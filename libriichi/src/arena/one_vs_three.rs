use super::game::{BatchGame, Index};
use super::result::GameResult;
use crate::agent::{AkochanAgent, BatchAgent, new_py_agent};
use crate::consts::{HANCHAN, NUM_PLAYERS, TONPUU};
use std::fs::{self, File};
use std::io;
use std::iter;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, ensure};
use flate2::Compression;
use flate2::read::GzEncoder;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct OneVsThree {
    pub disable_progress_bar: bool,
    pub log_dir: Option<String>,
    /// `HANCHAN` (8, or 6 in sanma) or `TONPUU` (4, or 3 in sanma).
    pub game_length: u8,
}

impl Default for OneVsThree {
    fn default() -> Self {
        Self {
            disable_progress_bar: false,
            log_dir: None,
            game_length: HANCHAN,
        }
    }
}

#[pymethods]
impl OneVsThree {
    #[new]
    #[pyo3(signature = (*, disable_progress_bar=false, log_dir=None, game_length=HANCHAN))]
    fn new(disable_progress_bar: bool, log_dir: Option<String>, game_length: u8) -> Result<Self> {
        ensure!(
            matches!(game_length, TONPUU | HANCHAN),
            "game_length must be {TONPUU} (tonpuu) or {HANCHAN} (hanchan), got {game_length}",
        );
        Ok(Self {
            disable_progress_bar,
            log_dir,
            game_length,
        })
    }

    /// Returns the rankings of the challenger.
    pub fn py_vs_py(
        &self,
        challenger: PyObject,
        champion: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<[i32; NUM_PLAYERS]> {
        // `allow_threads` is required, otherwise it will block python GC to
        // run, leading to memory leaks, since this function is doing long
        // tasks.
        py.allow_threads(move || {
            let results = self.run_batch(
                |player_ids| new_py_agent(challenger, player_ids),
                |player_ids| new_py_agent(champion, player_ids),
                seed_start,
                seed_count,
            )?;

            let mut rankings = [0; NUM_PLAYERS];
            for (i, result) in results.iter().enumerate() {
                let rank = result.rankings().rank_by_player[i % NUM_PLAYERS];
                rankings[rank as usize] += 1;
            }
            Ok(rankings)
        })
    }

    /// Returns the rankings of the challenger (akochan in this case).
    pub fn ako_vs_py(
        &self,
        engine: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<[i32; NUM_PLAYERS]> {
        py.allow_threads(move || {
            let results = self.run_batch(
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),
                |player_ids| new_py_agent(engine, player_ids),
                seed_start,
                seed_count,
            )?;

            let mut rankings = [0; NUM_PLAYERS];
            for (i, result) in results.iter().enumerate() {
                let rank = result.rankings().rank_by_player[i % NUM_PLAYERS];
                rankings[rank as usize] += 1;
            }
            Ok(rankings)
        })
    }

    /// Returns the rankings of the challenger (python agent in this case).
    pub fn py_vs_ako(
        &self,
        engine: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<[i32; NUM_PLAYERS]> {
        py.allow_threads(move || {
            let results = self.run_batch(
                |player_ids| new_py_agent(engine, player_ids),
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),
                seed_start,
                seed_count,
            )?;

            let mut rankings = [0; NUM_PLAYERS];
            for (i, result) in results.iter().enumerate() {
                let rank = result.rankings().rank_by_player[i % NUM_PLAYERS];
                rankings[rank as usize] += 1;
            }
            Ok(rankings)
        })
    }
}

impl OneVsThree {
    pub fn run_batch<C, M>(
        &self,
        new_challenger_agent: C,
        new_champion_agent: M,
        seed_start: (u64, u64),
        seed_count: u64,
    ) -> Result<Vec<GameResult>>
    where
        C: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
        M: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
    {
        if let Some(dir) = &self.log_dir {
            fs::create_dir_all(dir)?;
        }

        log::info!(
            "seed: [{}, {}) w/ {:#x}, start {} sets, {} games of {} hands",
            seed_start.0,
            seed_start.0 + seed_count,
            seed_start.1,
            seed_count,
            seed_count * NUM_PLAYERS as u64,
            self.game_length,
        );

        let seeds: Vec<_> = (seed_start.0..seed_start.0 + seed_count)
            .flat_map(|seed| iter::repeat_n((seed, seed_start.1), NUM_PLAYERS))
            .collect();

        let challenger_player_ids: Vec<_> = (0..NUM_PLAYERS as u8)
            .cycle()
            .take(seed_count as usize * NUM_PLAYERS)
            .collect();

        // In split k the challenger sits at seat k and the champion fills the
        // other seats.
        let champion_player_ids_per_seed: Vec<u8> = (0..NUM_PLAYERS as u8)
            .flat_map(|split| (0..NUM_PLAYERS as u8).filter(move |&id| id != split))
            .collect();
        let champion_player_ids: Vec<_> = champion_player_ids_per_seed
            .iter()
            .copied()
            .cycle()
            .take(seed_count as usize * champion_player_ids_per_seed.len())
            .collect();

        let mut agents = [
            new_challenger_agent(&challenger_player_ids)?,
            new_champion_agent(&champion_player_ids)?,
        ];
        let batch_game = BatchGame::tenhou(self.game_length, self.disable_progress_bar);

        let mut challenger_idx = 0;
        let mut champion_idx = 0;
        let agent_idxs_per_seed: Vec<[usize; NUM_PLAYERS]> = (0..NUM_PLAYERS)
            .map(|split| std::array::from_fn(|seat| usize::from(seat != split)))
            .collect();
        let indexes: Vec<_> = agent_idxs_per_seed
            .iter()
            .copied()
            .cycle()
            .take(seed_count as usize * agent_idxs_per_seed.len())
            .map(|agent_idxs_per_split| {
                agent_idxs_per_split.map(|agent_idx| {
                    let player_id_idx = if agent_idx == 0 {
                        &mut challenger_idx
                    } else {
                        &mut champion_idx
                    };
                    let ret = Index {
                        agent_idx,
                        player_id_idx: *player_id_idx,
                    };
                    *player_id_idx += 1;
                    ret
                })
            })
            .collect();

        let results = batch_game.run(&mut agents, &indexes, &seeds)?;

        if let Some(dir) = &self.log_dir {
            log::info!("dumping game logs");

            let bar = if self.disable_progress_bar {
                ProgressBar::hidden()
            } else {
                ProgressBar::new(seed_count * NUM_PLAYERS as u64)
            };
            const TEMPLATE: &str = "[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
            bar.set_style(ProgressStyle::with_template(TEMPLATE)?.progress_chars("#-"));
            bar.enable_steady_tick(Duration::from_millis(150));

            results
                .par_iter()
                .progress_with(bar)
                .enumerate()
                .try_for_each(|(i, game_result)| {
                    let split_name = ["a", "b", "c", "d"][i % NUM_PLAYERS];
                    let (seed, key) = game_result.seed;
                    let filename: PathBuf = [dir, &format!("{seed}_{key}_{split_name}.json.gz")]
                        .iter()
                        .collect();

                    let log = game_result.dump_json_log()?;
                    let mut comp = GzEncoder::new(log.as_bytes(), Compression::best());
                    let mut f = File::create(filename)?;
                    io::copy(&mut comp, &mut f)?;

                    anyhow::Ok(())
                })?;
        }

        Ok(results)
    }
}
