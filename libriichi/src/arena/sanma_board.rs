//! The three-player (sanma) board, compiled in place of `board.rs` when the
//! `sanma` feature is on. It keeps `board.rs`'s interface (`Board`,
//! `BoardState`, `Poll`, `AgentContext`) so `game.rs` and the 1v3 rotation
//! drive it unchanged.
//!
//! The rules follow Tenhou's phoenix-room sanma, which is what the training
//! data is. Each difference from the four-player board was checked against
//! 35,534 Tenhou sanma hanchan (321,963 hands):
//!
//! - 108 tiles (no 2m-8m, one red 5p and 5s), 13 x 3 dealt, a 14-tile dead
//!   wall and 55 live draws.
//! - No chi (the player states never offer it).
//! - Kita: a North set aside on one's own turn, then a replacement draw. An
//!   opponent may ron it. A kan dora still pending from an open or added kan
//!   is revealed right after the kita (289 cases in the logs, always in that
//!   order).
//! - Kans and kita share the replacement draws: the first four come from the
//!   dead wall, any after that from the far end of the live wall. Every
//!   replacement draw shortens the live wall by one.
//! - Each honba is worth 100 from each of the two other seats: 200 on a ron
//!   or a tsumo (checked on 9,549 rons and 10,087 tsumos).
//! - Tsumo loss: a tsumo is paid by the two other seats at four-player rates
//!   (`Point::tsumo_total`). Under pao the liable player pays that tsumo total
//!   in full.
//! - Abortive draws: nine terminals and four kans by more than one player
//!   only. Three riichi do not abort (1,748 hands), and four winds cannot
//!   happen with three seats.
//! - Double ron is allowed (1,125 in the logs); triple ron cannot happen.
//!
//! Oracle observations are not built for sanma; `BatchGame::run` refuses
//! agents that ask for them.

use super::result::KyokuResult;
use super::wall::SANMA_UNSHUFFLED;
use crate::consts::{LIVE_WALL, NOTEN_POT, NUM_PLAYERS, NUM_PLAYERS_U8};
use crate::mjai::{Event, EventExt};
use crate::state::PlayerState;
use crate::tile::Tile;
use crate::vec_ops::vec_add_assign;
use crate::{matches_tu8, must_tile, tu8};
use std::convert::TryInto;
use std::{array, mem};

use anyhow::{Context, Result, bail};
use derivative::Derivative;
use ndarray::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use sha3::{Digest, Sha3_256};
use tinyvec::ArrayVec;

/// Points one honba adds to a win: 100 from each other seat.
const HONBA_UNIT: i32 = 100 * (NUM_PLAYERS as i32 - 1);
/// Replacement tiles in the dead wall; replacements past these come from the
/// far end of the live wall.
const RINSHAN: usize = 4;
const DORA_INDICATORS: usize = 5;

/// The fields are all pub on purpose so the caller will be able to set the
/// yama, doras, scores directly.
///
/// See the module docs for how this differs from the four-player board.
#[derive(Debug, Default)]
pub struct Board {
    /// Counts from 0
    pub kyoku: u8,
    pub honba: u8,
    /// Does not effect the kyoku seed
    pub kyotaku: u8,
    /// [35000; NUM_PLAYERS]
    pub scores: [i32; NUM_PLAYERS],

    pub haipai: [[Tile; 13]; NUM_PLAYERS],
    /// Goes backward (pop); `yama[0]` is the haitei end, where replacement
    /// draws past the dead wall's four come from.
    pub yama: Vec<Tile>,
    /// Goes backward (pop)
    pub rinshan: Vec<Tile>,
    /// Goes backward (pop)
    pub dora_indicators: Vec<Tile>,
    /// Goes forward (iter)
    pub ura_indicators: Vec<Tile>,
}

#[derive(Derivative)]
#[derivative(Default)]
pub struct BoardState {
    board: Board,
    // Absolute seat, with the oya of E1 always being 0
    oya: u8,
    player_states: [PlayerState; NUM_PLAYERS],

    can_renchan: bool,
    has_hora: bool,
    has_abortive_ryukyoku: bool,
    kyoku_deltas: [i32; NUM_PLAYERS],

    #[derivative(Default(value = "LIVE_WALL"))]
    tiles_left: u8,
    tsumo_actor: u8,
    // Just a fancy bool
    deal_from_rinshan: Option<()>,
    need_new_dora_at_discard: Option<()>,
    need_new_dora_at_tsumo: Option<()>,
    riichi_to_be_accepted: Option<u8>,
    #[derivative(Default(value = "[true; NUM_PLAYERS]"))]
    can_nagashi_mangan: [bool; NUM_PLAYERS],
    kans: u8,
    check_four_kan: bool,
    paos: [Option<u8>; NUM_PLAYERS],

    log: Vec<EventExt>,
}

pub struct AgentContext<'a> {
    pub player_states: &'a [PlayerState; NUM_PLAYERS],
    pub log: &'a [EventExt],
}

#[derive(Clone, Copy)]
pub enum Poll {
    InGame,
    End,
}

impl Board {
    pub fn init_from_seed(&mut self, game_seed: (u64, u64)) {
        let (nonce, key) = game_seed;
        let kyoku_seed = Sha3_256::new()
            .chain_update(nonce.to_le_bytes())
            .chain_update(key.to_le_bytes())
            .chain_update([self.kyoku, self.honba])
            .finalize()
            .into();
        let mut rng = ChaCha12Rng::from_seed(kyoku_seed);
        let mut seq = SANMA_UNSHUFFLED;
        seq.shuffle(&mut rng);

        self.haipai = array::from_fn(|i| seq[i * 13..(i + 1) * 13].try_into().unwrap());
        let mut idx = 13 * NUM_PLAYERS;

        self.rinshan = seq[idx..idx + RINSHAN].to_vec();
        idx += RINSHAN;
        self.dora_indicators = seq[idx..idx + DORA_INDICATORS].to_vec();
        idx += DORA_INDICATORS;
        self.ura_indicators = seq[idx..idx + DORA_INDICATORS].to_vec();
        idx += DORA_INDICATORS;
        self.yama = seq[idx..idx + LIVE_WALL as usize].to_vec();
        idx += LIVE_WALL as usize;
        assert_eq!(idx, seq.len());
    }

    /// `game_length` is 6 for hanchan and 3 for tonpuu; the players' states
    /// need it to tell which round is all-last.
    pub fn into_state(self, game_length: u8) -> BoardState {
        let oya = self.kyoku % NUM_PLAYERS_U8;

        BoardState {
            board: self,
            oya,
            player_states: array::from_fn(|i| PlayerState::new_with_length(i as u8, game_length)),
            ..Default::default()
        }
    }
}

impl BoardState {
    /// Returns iff any player on the board can act or the kyoku has ended.
    pub fn poll(&mut self, mut reactions: [EventExt; NUM_PLAYERS]) -> Result<Poll> {
        loop {
            let poll = self.step(&reactions)?;
            match poll {
                Poll::InGame => {
                    if self.player_states.iter().any(|c| c.last_cans().can_act()) {
                        return Ok(poll);
                    }
                }
                Poll::End => {
                    self.add_log_no_meta(Event::EndKyoku);
                    vec_add_assign(&mut self.board.scores, &self.kyoku_deltas);
                    if self.has_abortive_ryukyoku {
                        self.can_renchan = true;
                    }
                    return Ok(poll);
                }
            };
            reactions = Default::default();
        }
    }

    #[inline]
    pub fn agent_context(&self) -> AgentContext<'_> {
        AgentContext {
            player_states: &self.player_states,
            log: &self.log,
        }
    }

    #[inline]
    pub const fn end(&self) -> KyokuResult {
        KyokuResult {
            kyoku: self.board.kyoku,
            can_renchan: self.can_renchan,
            has_hora: self.has_hora,
            has_abortive_ryukyoku: self.has_abortive_ryukyoku,
            kyotaku_left: self.board.kyotaku,
            scores: self.board.scores,
        }
    }

    #[inline]
    pub fn take_log(&mut self) -> Vec<EventExt> {
        mem::take(&mut self.log)
    }

    #[inline]
    fn add_log(&mut self, ev: EventExt) {
        self.log.push(ev);
    }

    #[inline]
    fn add_log_no_meta(&mut self, ev: Event) {
        self.log.push(EventExt::no_meta(ev));
    }

    #[inline]
    fn broadcast(&mut self, ev: &Event) {
        for s in &mut self.player_states {
            s.update(ev).expect("fatal internal bug in BoardState");
        }
    }

    fn haipai(&mut self) -> Result<()> {
        let bakaze = must_tile!(tu8!(E) + self.board.kyoku / NUM_PLAYERS_U8);
        let start_kyoku = Event::StartKyoku {
            bakaze,
            dora_marker: self
                .board
                .dora_indicators
                .pop()
                .context("insufficient dora indicators")?,
            kyoku: self.oya + 1,
            honba: self.board.honba,
            kyotaku: self.board.kyotaku,
            oya: self.oya,
            scores: self.board.scores,
            tehais: self.board.haipai,
        };
        self.broadcast(&start_kyoku);
        self.add_log_no_meta(start_kyoku);

        let tile = self
            .board
            .yama
            .pop()
            .context("invalid yama: empty at init")?;
        self.tiles_left -= 1;
        let first_tsumo = Event::Tsumo {
            actor: self.oya,
            pai: tile,
        };
        self.broadcast(&first_tsumo);
        self.add_log_no_meta(first_tsumo);

        Ok(())
    }

    fn exhaustive_ryukyoku(&mut self) {
        let mut deltas = [0; NUM_PLAYERS];
        self.can_renchan = self.player_states[self.oya as usize].shanten() == 0;

        // Tsumo loss: a nagashi mangan is paid like a mangan tsumo by the two
        // other seats, with no honba (dealer 8000, non-dealer 6000).
        let mut has_nagashi_mangan = false;
        self.can_nagashi_mangan
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b)
            .map(|(i, _)| i)
            .for_each(|i| {
                has_nagashi_mangan = true;
                if i as u8 == self.oya {
                    let mut dod = [-4000; NUM_PLAYERS];
                    dod[i] = 4000 * (NUM_PLAYERS as i32 - 1);
                    vec_add_assign(&mut deltas, &dod);
                } else {
                    let mut dod = [-2000; NUM_PLAYERS];
                    dod[i] = 2000 * (NUM_PLAYERS as i32 - 2) + 4000;
                    dod[self.oya as usize] = -4000;
                    vec_add_assign(&mut deltas, &dod);
                };
            });

        if !has_nagashi_mangan {
            let tenpai_actors: ArrayVec<[_; NUM_PLAYERS]> = self
                .player_states
                .iter()
                .enumerate()
                .filter(|&(_, s)| s.shanten() == 0)
                .map(|(i, _)| i)
                .collect();

            // The pot is split among the tenpai players and paid by the rest;
            // nothing changes hands if everyone or no one is tenpai.
            let tenpai = tenpai_actors.len() as i32;
            let (plus, minus) = if 0 < tenpai && tenpai < NUM_PLAYERS as i32 {
                (
                    NOTEN_POT / tenpai,
                    -NOTEN_POT / (NUM_PLAYERS as i32 - tenpai),
                )
            } else {
                (0, 0)
            };
            if plus > 0 {
                let mut dod = [minus; NUM_PLAYERS];
                tenpai_actors.into_iter().for_each(|i| dod[i] = plus);
                vec_add_assign(&mut deltas, &dod);
            }
        }

        vec_add_assign(&mut self.kyoku_deltas, &deltas);
        let ryukyoku = Event::Ryukyoku {
            deltas: Some(deltas),
        };
        self.add_log_no_meta(ryukyoku);
        // no need to broadcast
    }

    const fn update_nagashi_mangan(&mut self, ev: &Event) {
        match *ev {
            Event::Dahai { actor, pai, .. } if !pai.is_yaokyuu() => {
                self.can_nagashi_mangan[actor as usize] = false;
            }
            Event::Pon { target, .. } | Event::Daiminkan { target, .. } => {
                self.can_nagashi_mangan[target as usize] = false;
            }
            _ => (),
        };
    }

    fn check_riichi_accepted(&mut self) {
        if let Some(actor) = self.riichi_to_be_accepted.take() {
            let riichi_accepted = Event::ReachAccepted { actor };
            self.broadcast(&riichi_accepted);
            self.add_log_no_meta(riichi_accepted);
            self.board.scores[actor as usize] -= 1000;
            self.board.kyotaku += 1;
        }
    }

    fn add_new_dora(&mut self) -> Result<()> {
        let dora = self
            .board
            .dora_indicators
            .pop()
            .context("illegal kan: already 4 kans and this is the 5th")?;
        let dora_ev = Event::Dora { dora_marker: dora };
        self.broadcast(&dora_ev);
        self.add_log_no_meta(dora_ev);

        Ok(())
    }

    /// The next replacement draw, for a kan or a kita: from the dead wall
    /// while it lasts, then from the haitei end of the live wall.
    fn draw_replacement(&mut self) -> Result<Tile> {
        if let Some(tile) = self.board.rinshan.pop() {
            return Ok(tile);
        }
        if self.board.yama.is_empty() {
            bail!("illegal replacement draw: the dead wall and the live wall are empty");
        }
        Ok(self.board.yama.remove(0))
    }

    fn handle_hora(
        &mut self,
        single_actor: u8,
        single_target: u8,
        reactions: &[EventExt; NUM_PLAYERS],
    ) -> Result<()> {
        self.has_hora = true;

        let is_ron = single_actor != single_target;
        let mut honba_left = self.board.honba as i32; // mut in case of multi-ron
        let mut kyotaku_point = self.board.kyotaku as i32 * 1000; // ditto
        self.board.kyotaku = 0; // Unlike honba, kyotaku in self will be cleared

        // Let the states get their agari points provided with our ura
        // indicators.
        let ura_indicators = self.board.ura_indicators
            [..DORA_INDICATORS - self.board.dora_indicators.len()]
            .to_vec();
        let points = reactions
            .iter()
            .map(|ev| match ev.event {
                Event::Hora { actor, .. } => {
                    self.can_renchan |= actor == self.oya;
                    let point =
                        self.player_states[actor as usize].agari_points(is_ron, &ura_indicators);
                    Some(point).transpose()
                }
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;

        if is_ron {
            // Double ron: the first winner after the target, in turn order,
            // takes the honba and the riichi sticks.
            points
                .into_iter()
                .enumerate()
                .cycle()
                .skip(single_target as usize + 1)
                .take(NUM_PLAYERS - 1)
                .filter_map(|(actor, v)| v.map(|point| (actor, point)))
                .for_each(|(actor, point)| {
                    let mut deltas = [0; NUM_PLAYERS];
                    if let Some(pao_target) = self.paos[actor] {
                        // As per [Tenhou's rule](https://tenhou.net/man/#RULE):
                        //
                        // > 複合役満を含む得点を、ツモ＝全額・ロン＝折半で支払
                        // > う。積み棒は包。
                        deltas[pao_target as usize] = -point.ron / 2 - honba_left * HONBA_UNIT;
                        deltas[single_target as usize] -= point.ron / 2; // they may be the same person
                    } else {
                        deltas[single_target as usize] = -point.ron - honba_left * HONBA_UNIT;
                    }
                    deltas[actor] = point.ron + kyotaku_point + honba_left * HONBA_UNIT;

                    kyotaku_point = 0;
                    honba_left = 0;

                    vec_add_assign(&mut self.kyoku_deltas, &deltas);
                    let ura_markers = if self.player_states[actor].self_riichi_accepted() {
                        ura_indicators.clone()
                    } else {
                        Default::default()
                    };

                    let hora = Event::Hora {
                        actor: actor as u8,
                        target: single_target,
                        deltas: Some(deltas),
                        ura_markers: Some(ura_markers),
                    };
                    self.add_log_no_meta(hora);
                    // No need to broadcast
                });
            return Ok(());
        }

        let point = points[single_actor as usize].unwrap();
        let is_oya = single_actor == self.oya;
        let mut deltas = [0; NUM_PLAYERS];
        if let Some(pao_target) = self.paos[single_actor as usize] {
            // The liable player pays the whole tsumo, which tsumo loss makes
            // smaller than the ron value (24000 for a non-dealer yakuman).
            deltas[pao_target as usize] = -point.tsumo_total(is_oya) - honba_left * HONBA_UNIT;
        } else {
            deltas.fill(-point.tsumo_ko - honba_left * 100);
            if !is_oya {
                deltas[self.oya as usize] = -point.tsumo_oya - honba_left * 100;
            }
        };
        deltas[single_actor as usize] =
            point.tsumo_total(is_oya) + kyotaku_point + honba_left * HONBA_UNIT;

        vec_add_assign(&mut self.kyoku_deltas, &deltas);
        let ura_markers = if self.player_states[single_actor as usize].self_riichi_accepted() {
            ura_indicators
        } else {
            Default::default()
        };

        let hora = Event::Hora {
            actor: single_actor,
            target: single_target,
            deltas: Some(deltas),
            ura_markers: Some(ura_markers),
        };
        self.add_log_no_meta(hora);
        // No need to broadcast

        Ok(())
    }

    fn update_paos(&mut self, ev: &Event) {
        match *ev {
            Event::Pon {
                target, actor, pai, ..
            }
            | Event::Daiminkan {
                target, actor, pai, ..
            } if pai.is_jihai() => {
                let mut jihais = 0_u8;
                self.player_states[actor as usize]
                    .pons()
                    .iter()
                    .chain(self.player_states[actor as usize].minkans())
                    .copied()
                    .filter(|&t| t >= tu8!(E))
                    .for_each(|t| jihais |= 1 << (t - tu8!(E)));
                let daisanein_confirmed = (jihais & 0b1110000) == 0b1110000;
                let daisuushi_confirmed = (jihais & 0b0001111) == 0b0001111;
                if daisanein_confirmed && matches_tu8!(pai.as_u8(), P | F | C)
                    || daisuushi_confirmed && matches_tu8!(pai.as_u8(), E | S | W | N)
                {
                    self.paos[actor as usize] = Some(target);
                }
            }
            _ => (),
        }
    }

    #[inline]
    fn abortive_ryukyoku(&mut self) {
        let ryukyoku = Event::Ryukyoku {
            deltas: Some([0; NUM_PLAYERS]),
        };
        self.add_log_no_meta(ryukyoku);
        self.has_abortive_ryukyoku = true;
        // No need to broadcast
    }

    fn step(&mut self, reactions: &[EventExt; NUM_PLAYERS]) -> Result<Poll> {
        if self.tiles_left == LIVE_WALL {
            self.haipai()?;
            return Ok(Poll::InGame);
        }

        // Validate reactions
        for (actor, ev) in reactions.iter().enumerate() {
            self.player_states[actor]
                .validate_reaction(&ev.event)
                .with_context(|| {
                    format!(
                        "invalid action: {ev:?}\nstate:\n{}",
                        self.player_states[actor].brief_info(),
                    )
                })?;
        }

        let ev = reactions
            .iter()
            .min_by_key(|ev| match ev.event {
                Event::Hora { .. } => 0,
                Event::Daiminkan { .. } | Event::Pon { .. } => 1,
                Event::None => 3,
                _ => 2,
            })
            .unwrap(); // Unwrap is safe because it is proven non-empty

        if self.check_four_kan && !matches!(ev.event, Event::Hora { .. }) {
            // 四槓散了
            self.abortive_ryukyoku();
            return Ok(Poll::End);
        }

        self.update_nagashi_mangan(&ev.event);

        match ev.event {
            Event::None => {
                if self.tiles_left == 0 {
                    self.exhaustive_ryukyoku();
                    return Ok(Poll::End);
                }
                self.check_riichi_accepted();

                let tile = if self.deal_from_rinshan.take().is_some() {
                    self.draw_replacement()?
                } else {
                    self.board.yama.pop().with_context(|| {
                        format!("tiles left > 0 ({}) but yama is empty", self.tiles_left)
                    })?
                };
                self.tiles_left -= 1;
                let tsumo = Event::Tsumo {
                    actor: self.tsumo_actor,
                    pai: tile,
                };

                // This is for Kakan only because chankan is possible until an
                // actual tsumo.
                if self.need_new_dora_at_tsumo.take().is_some() {
                    self.add_new_dora()?;
                }

                self.broadcast(&tsumo);
                self.add_log_no_meta(tsumo);
            }

            Event::Dahai { actor, .. } => {
                if self.need_new_dora_at_discard.take().is_some() {
                    self.add_new_dora()?;
                }

                self.broadcast(&ev.event);
                self.add_log(ev.clone());
                self.tsumo_actor = (actor + 1) % NUM_PLAYERS_U8;

                if self.kans == 4 && self.player_states.iter().all(|s| s.kans_count() < 4) {
                    // 四槓散了
                    self.check_four_kan = true;
                }
            }

            Event::Pon { .. } => {
                self.check_riichi_accepted();
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
            }

            Event::Ankan { actor, .. } => {
                // For continuous kan
                if self.need_new_dora_at_discard.take().is_some() {
                    self.add_new_dora()?;
                }

                self.broadcast(&ev.event);
                self.add_log(ev.clone());

                // Immediately add new dora
                self.add_new_dora()?;

                self.tsumo_actor = actor;
                self.deal_from_rinshan = Some(());
                self.kans += 1;
            }

            Event::Daiminkan { actor, .. } | Event::Kakan { actor, .. } => {
                // For Kakan only, do not `.take()` it.
                if self.need_new_dora_at_discard.is_some() {
                    self.need_new_dora_at_tsumo = Some(());
                }

                // For Daiminkan only
                self.check_riichi_accepted();

                self.broadcast(&ev.event);
                self.add_log(ev.clone());

                self.need_new_dora_at_discard = Some(());

                self.tsumo_actor = actor;
                self.deal_from_rinshan = Some(());
                self.kans += 1;
            }

            Event::Nukidora { actor, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());

                // A kan dora still pending from an open or added kan is
                // revealed right after the kita, as in Tenhou's logs.
                if self.need_new_dora_at_discard.take().is_some() {
                    self.add_new_dora()?;
                }

                // Opponents may ron the North; if nobody does, the actor draws
                // a replacement.
                self.tsumo_actor = actor;
                self.deal_from_rinshan = Some(());
            }

            Event::Reach { actor } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
                self.riichi_to_be_accepted = Some(actor);
            }

            Event::Hora { actor, target, .. } => {
                self.handle_hora(actor, target, reactions)?;
                return Ok(Poll::End);
            }

            Event::Ryukyoku { .. } => {
                // 九種九牌
                self.abortive_ryukyoku();
                return Ok(Poll::End);
            }

            _ => {
                bail!("unexpected event: {:?}", ev.event);
            }
        };

        // The pao check cannot be done before the current event (Pon or
        // Daiminkan) gets processed because it requires to read `.pons()` and
        // `.minkans()`.
        self.update_paos(&ev.event);

        Ok(Poll::InGame)
    }

    /// Not built for sanma: `BatchGame::run` refuses oracle agents first.
    pub fn encode_oracle_obs(&self, _perspective: u8, _version: u32) -> Array2<f32> {
        unreachable!("sanma has no oracle observation")
    }
}

#[cfg(test)]
mod test {
    use super::super::game::{BatchGame, Index};
    use super::super::result::GameResult;
    use crate::agent::{Agent, BatchAgent, BatchifiedAgent, InvisibleState};
    use crate::consts::{INIT_SCORE, NUM_PLAYERS, NUM_PLAYERS_U8};
    use crate::mjai::{Event, EventExt};
    use crate::stat::Stat;
    use crate::state::PlayerState;
    use crate::tile::Tile;
    use crate::{must_tile, t, tu8};
    use std::array;
    use std::collections::HashMap;

    use anyhow::{Context, Result};
    use rand::prelude::*;
    use rand_chacha::ChaCha12Rng;

    /// Plays every sanma action at random so the tests reach kita (and ron
    /// on it), kans, pao, riichi and double ron: it always wins when it can,
    /// and otherwise sets a North aside, kans, pons and riichis with fixed
    /// odds, discarding into tenpai when it can and else at random (honors
    /// and terminals first).
    struct RandomSanma {
        id: u8,
        rng: ChaCha12Rng,
    }

    impl RandomSanma {
        fn new_batched(player_ids: &[u8], seed: u64) -> Result<BatchifiedAgent<Self>> {
            let mut n = 0;
            BatchifiedAgent::new(
                |id| {
                    n += 1;
                    Ok(Self {
                        id,
                        rng: ChaCha12Rng::seed_from_u64(seed * 1000 + n),
                    })
                },
                player_ids,
            )
        }

        fn discard(&mut self, state: &PlayerState) -> Event {
            let tenpai = state.discard_candidates_with_unconditional_tenpai_aka();
            let legal = state.discard_candidates_aka();
            let pick = |set: &[bool; 37]| -> Vec<Tile> {
                (0..37).filter(|&i| set[i]).map(|i| must_tile!(i)).collect()
            };
            let mut candidates: Vec<_> = pick(&tenpai)
                .into_iter()
                .filter(|t| legal[t.as_usize()])
                .collect();
            if candidates.is_empty() {
                let all = pick(&legal);
                let yaokyuu: Vec<_> = all.iter().copied().filter(|t| t.is_yaokyuu()).collect();
                candidates = if !yaokyuu.is_empty() && self.rng.random_bool(0.7) {
                    yaokyuu
                } else {
                    all
                };
            }
            let pai = candidates[self.rng.random_range(0..candidates.len())];
            Event::Dahai {
                actor: self.id,
                pai,
                tsumogiri: state.last_self_tsumo() == Some(pai),
            }
        }
    }

    impl Agent for RandomSanma {
        fn name(&self) -> String {
            "random-sanma".to_owned()
        }

        fn react(
            &mut self,
            _: &[EventExt],
            state: &PlayerState,
            _: Option<InvisibleState>,
        ) -> Result<EventExt> {
            let cans = state.last_cans();
            let actor = self.id;
            let akas = state.akas_in_hand();
            let can_akaize = |tile: Tile| match tile.deaka().as_u8() {
                tu8!(5m) => akas[0],
                tu8!(5p) => akas[1],
                tu8!(5s) => akas[2],
                _ => false,
            };
            let ev = if cans.can_agari() {
                Event::Hora {
                    actor,
                    target: cans.target_actor,
                    deltas: None,
                    ura_markers: None,
                }
            } else if cans.can_kita && self.rng.random_bool(0.9) {
                Event::Nukidora { actor, pai: t!(N) }
            } else if cans.can_ryukyoku && self.rng.random_bool(0.5) {
                Event::Ryukyoku { deltas: None }
            } else if cans.can_daiminkan && self.rng.random_bool(0.5) {
                let tile = state.last_kawa_tile().context("no last kawa tile")?;
                let consumed = if tile.is_aka() {
                    [tile.deaka(); 3]
                } else {
                    [tile.akaize(), tile, tile]
                };
                Event::Daiminkan {
                    actor,
                    target: cans.target_actor,
                    pai: tile,
                    consumed,
                }
            } else if cans.can_pon && self.rng.random_bool(0.4) {
                let pai = state.last_kawa_tile().context("no last kawa tile")?;
                let consumed = if can_akaize(pai) && !pai.is_aka() {
                    [pai.akaize(), pai.deaka()]
                } else {
                    [pai.deaka(); 2]
                };
                Event::Pon {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            } else if cans.can_ankan && self.rng.random_bool(0.6) {
                let tile = state.ankan_candidates()[0];
                Event::Ankan {
                    actor,
                    consumed: [tile.akaize(), tile, tile, tile],
                }
            } else if cans.can_kakan && self.rng.random_bool(0.6) {
                let tile = state.kakan_candidates()[0];
                let (pai, consumed) = if can_akaize(tile) {
                    (tile.akaize(), [tile.deaka(); 3])
                } else {
                    (tile.deaka(), [tile.akaize(), tile.deaka(), tile.deaka()])
                };
                Event::Kakan {
                    actor,
                    pai,
                    consumed,
                }
            } else if cans.can_riichi && self.rng.random_bool(0.7) {
                Event::Reach { actor }
            } else if cans.can_discard {
                self.discard(state)
            } else {
                Event::None
            };
            Ok(EventExt::no_meta(ev))
        }
    }

    fn play(g: &BatchGame, seed_count: u64, agent_seed: u64) -> Vec<GameResult> {
        let player_ids: Vec<_> = (0..NUM_PLAYERS_U8)
            .cycle()
            .take(seed_count as usize * NUM_PLAYERS)
            .collect();
        let mut agents: [Box<dyn BatchAgent>; 1] = [Box::new(
            RandomSanma::new_batched(&player_ids, agent_seed).unwrap(),
        )];
        let indexes: Vec<_> = (0..seed_count as usize)
            .map(|game| {
                array::from_fn(|i| Index {
                    agent_idx: 0,
                    player_id_idx: game * NUM_PLAYERS + i,
                })
            })
            .collect();
        let seeds: Vec<_> = (0..seed_count).map(|s| (s, agent_seed)).collect();
        g.run(&mut agents, &indexes, &seeds).unwrap()
    }

    /// Replays a game's log into three fresh player states (each checks
    /// every event against the rules, as the training loader does) and
    /// checks the points: within a hand, the win or draw deltas add up to
    /// the riichi sticks they collect, and each hand starts from the last
    /// one's scores plus its deltas less its riichi deposits.
    fn check_log(result: &GameResult, counts: &mut HashMap<&'static str, usize>) {
        let mut states: [PlayerState; NUM_PLAYERS] = array::from_fn(|i| PlayerState::new(i as u8));
        let mut expected_scores = [INIT_SCORE; NUM_PLAYERS];
        let mut kyotaku_at_start = 0;
        for kyoku in &result.game_log {
            let mut riichis = [0; NUM_PLAYERS];
            let mut collected = false;
            let mut horas = 0;
            let mut after_kita = false;
            for ev in kyoku {
                match &ev.event {
                    Event::StartKyoku {
                        scores, kyotaku, ..
                    } => {
                        assert_eq!(*scores, expected_scores, "seed {:?}", result.seed);
                        kyotaku_at_start = *kyotaku as i32;
                    }
                    Event::ReachAccepted { actor } => riichis[*actor as usize] += 1,
                    Event::Nukidora { .. } => *counts.entry("nukidora").or_default() += 1,
                    Event::Pon { .. } => *counts.entry("pon").or_default() += 1,
                    Event::Ankan { .. } => *counts.entry("ankan").or_default() += 1,
                    Event::Kakan { .. } => *counts.entry("kakan").or_default() += 1,
                    Event::Daiminkan { .. } => *counts.entry("daiminkan").or_default() += 1,
                    Event::Chi { .. } => panic!("chi in sanma"),
                    _ => (),
                }
                match &ev.event {
                    Event::Hora {
                        actor,
                        target,
                        deltas,
                        ..
                    } => {
                        horas += 1;
                        let d = deltas.unwrap();
                        let sum: i32 = d.iter().sum();
                        // only the first winner collects the sticks
                        let sticks = if collected {
                            0
                        } else {
                            (kyotaku_at_start + riichis.iter().sum::<i32>()) * 1000
                        };
                        collected = true;
                        assert_eq!(sum, sticks, "seed {:?}: hora deltas {d:?}", result.seed);
                        for (i, x) in d.iter().enumerate() {
                            expected_scores[i] += x;
                        }
                        let key = if actor == target { "tsumo" } else { "ron" };
                        *counts.entry(key).or_default() += 1;
                        if after_kita {
                            *counts.entry("ron_on_kita").or_default() += 1;
                        }
                        if horas == 2 {
                            *counts.entry("double_ron").or_default() += 1;
                        }
                    }
                    Event::Ryukyoku { deltas } => {
                        let d = deltas.unwrap();
                        assert_eq!(d.iter().sum::<i32>(), 0, "seed {:?}", result.seed);
                        let key = if d == [0; NUM_PLAYERS] {
                            "ryukyoku_even"
                        } else {
                            "ryukyoku_paid"
                        };
                        *counts.entry(key).or_default() += 1;
                        for (i, x) in d.iter().enumerate() {
                            expected_scores[i] += x;
                        }
                    }
                    // the training loader skips these, the states do not need them
                    Event::EndKyoku => continue,
                    _ => (),
                }
                if !matches!(ev.event, Event::Hora { .. }) {
                    after_kita = matches!(ev.event, Event::Nukidora { .. });
                }
                if !matches!(ev.event, Event::Hora { .. } | Event::Ryukyoku { .. }) {
                    for s in &mut states {
                        s.update(&ev.event).unwrap_or_else(|e| {
                            panic!("seed {:?}: {e:#} at {:?}", result.seed, ev.event)
                        });
                    }
                }
            }
            for i in 0..NUM_PLAYERS {
                expected_scores[i] -= riichis[i] * 1000;
            }
        }

        // The sticks left on the table go to the top when the game ends.
        let total: i32 = result.scores.iter().sum();
        assert_eq!(
            total,
            INIT_SCORE * NUM_PLAYERS as i32,
            "seed {:?}",
            result.seed
        );

        // `Stat`, which the evaluation scripts read the dumped logs with,
        // agrees with the arena on every seat's place and points.
        let log = result.dump_json_log().unwrap();
        let rankings = result.rankings();
        for seat in 0..NUM_PLAYERS {
            let stat = Stat::from_log(&log, seat as u8).unwrap();
            let mut place = [0; NUM_PLAYERS];
            place[rankings.rank_by_player[seat] as usize] = 1;
            assert_eq!(
                stat.rank_counts(),
                place,
                "seed {:?} seat {seat}",
                result.seed
            );
            assert_eq!(
                stat.point,
                (result.scores[seat] - INIT_SCORE) as i64,
                "seed {:?} seat {seat}",
                result.seed,
            );
        }
    }

    #[test]
    fn random_hanchan_follow_the_rules() {
        let mut counts = HashMap::new();
        let mut hands = 0;
        for agent_seed in 0..4 {
            for result in play(&BatchGame::tenhou_hanchan(true), 100, agent_seed) {
                hands += result.game_log.len();
                assert!(result.game_log.iter().all(|k| matches!(
                    k[0].event,
                    Event::StartKyoku { tehais, .. } if tehais.iter().flatten().all(|t| !t.is_sanma_removed())
                )));
                check_log(&result, &mut counts);
            }
        }
        println!("{hands} hands: {counts:?}");
        for key in [
            "nukidora",
            "pon",
            "ankan",
            "kakan",
            "daiminkan",
            "tsumo",
            "ron",
            "double_ron",
            "ryukyoku_even",
            "ryukyoku_paid",
        ] {
            assert!(
                counts.get(key).copied().unwrap_or(0) > 0,
                "never saw {key}: {counts:?}"
            );
        }
    }

    #[test]
    fn random_tonpuu_follow_the_rules() {
        let mut counts = HashMap::new();
        for result in play(&BatchGame::tenhou_tonpuu(true), 100, 7) {
            check_log(&result, &mut counts);
        }
    }

    /// Hora when possible, kita when possible, else tsumogiri.
    struct KitaFirst(u8);

    impl Agent for KitaFirst {
        fn name(&self) -> String {
            "kita-first".to_owned()
        }

        fn react(
            &mut self,
            _: &[EventExt],
            state: &PlayerState,
            _: Option<InvisibleState>,
        ) -> Result<EventExt> {
            let cans = state.last_cans();
            let ev = if cans.can_agari() {
                Event::Hora {
                    actor: self.0,
                    target: cans.target_actor,
                    deltas: None,
                    ura_markers: None,
                }
            } else if cans.can_kita {
                Event::Nukidora {
                    actor: self.0,
                    pai: t!(N),
                }
            } else if cans.can_discard {
                Event::Dahai {
                    actor: self.0,
                    pai: state.last_self_tsumo().context("no last tsumo")?,
                    tsumogiri: true,
                }
            } else {
                Event::None
            };
            Ok(EventExt::no_meta(ev))
        }
    }

    /// The dealer draws a North and sets it aside; the next seat, waiting on
    /// North with 123456789p PPP, rons it. The North is not a discard, so the
    /// ron is paid by the kita's owner like any ron, and the replacement
    /// draw never happens.
    #[test]
    fn ron_on_kita() {
        let mut pool = super::SANMA_UNSHUFFLED.to_vec();
        let mut take = |tile: Tile| {
            let i = pool.iter().position(|&t| t == tile).unwrap();
            pool.remove(i)
        };
        let waiting: [Tile; 13] = [
            t!(1p),
            t!(2p),
            t!(3p),
            t!(4p),
            t!(5p),
            t!(6p),
            t!(7p),
            t!(8p),
            t!(9p),
            t!(P),
            t!(P),
            t!(P),
            t!(N),
        ]
        .map(&mut take);
        let first_draw = take(t!(N));
        // the dealer and the third seat get no North, so no one else kitas
        let mut rest: Vec<_> = pool
            .iter()
            .copied()
            .filter(|t| t.deaka() != t!(N))
            .collect();
        let norths: Vec<_> = pool
            .iter()
            .copied()
            .filter(|t| t.deaka() == t!(N))
            .collect();
        let dealer: [Tile; 13] = rest.drain(..13).collect::<Vec<_>>().try_into().unwrap();
        let third: [Tile; 13] = rest.drain(..13).collect::<Vec<_>>().try_into().unwrap();
        rest.extend(norths);
        let rinshan = rest.drain(..4).collect();
        let dora_indicators = rest.drain(..5).collect();
        let ura_indicators = rest.drain(..5).collect();
        let mut yama = rest;
        yama.push(first_draw); // popped first
        assert_eq!(yama.len(), 55);

        let board = super::Board {
            scores: [INIT_SCORE; NUM_PLAYERS],
            honba: 1,
            haipai: [dealer, waiting, third],
            yama,
            rinshan,
            dora_indicators,
            ura_indicators,
            ..Default::default()
        };
        let mut state = board.into_state(6);
        let mut agents: [KitaFirst; NUM_PLAYERS] = array::from_fn(|i| KitaFirst(i as u8));
        let mut reactions: [EventExt; NUM_PLAYERS] = Default::default();
        loop {
            match state.poll(reactions).unwrap() {
                super::Poll::End => break,
                super::Poll::InGame => {
                    let ctx = state.agent_context();
                    reactions = array::from_fn(|i| {
                        let s = &ctx.player_states[i];
                        if s.last_cans().can_act() {
                            agents[i].react(ctx.log, s, None).unwrap()
                        } else {
                            EventExt::no_meta(Event::None)
                        }
                    });
                }
            }
        }
        let log: Vec<_> = state.take_log().into_iter().map(|e| e.event).collect();
        let kita = log
            .iter()
            .position(|e| matches!(e, Event::Nukidora { actor: 0, .. }))
            .expect("the dealer never set the North aside");
        match log[kita + 1] {
            Event::Hora {
                actor: 1,
                target: 0,
                deltas: Some(d),
                ..
            } => {
                assert_eq!(d.iter().sum::<i32>(), 0);
                assert_eq!(d[2], 0, "the third seat does not pay a ron");
                // the honba is 200 on a ron in sanma
                assert_eq!((d[1] - 200) % 1000, 0, "ron with one honba: {d:?}");
            }
            ref e => panic!("expected a ron on the kita, got {e:?}"),
        }
    }

    #[test]
    fn wall_layout() {
        let mut board = super::Board::default();
        board.init_from_seed((1, 2));
        assert_eq!(board.yama.len(), 55);
        assert_eq!(board.rinshan.len(), 4);
        assert_eq!(board.dora_indicators.len(), 5);
        assert_eq!(board.ura_indicators.len(), 5);
        let mut all: Vec<_> = board.haipai.iter().flatten().copied().collect();
        all.extend(&board.yama);
        all.extend(&board.rinshan);
        all.extend(&board.dora_indicators);
        all.extend(&board.ura_indicators);
        assert_eq!(all.len(), 108);
        assert!(all.iter().all(|t| !t.is_sanma_removed()));
        assert_eq!(all.iter().filter(|t| t.deaka() == t!(N)).count(), 4);
    }

    /// The round wind follows three hands a wind: S1 is the fourth hand.
    #[test]
    fn bakaze_changes_every_three_hands() {
        for (kyoku, wind) in [(0, t!(E)), (2, t!(E)), (3, t!(S)), (5, t!(S)), (6, t!(W))] {
            let mut board = super::Board {
                kyoku,
                scores: [INIT_SCORE; NUM_PLAYERS],
                ..Default::default()
            };
            board.init_from_seed((0, 0));
            let mut state = board.into_state(6);
            state.haipai().unwrap();
            match state.log[0].event {
                Event::StartKyoku {
                    bakaze,
                    kyoku: k,
                    oya,
                    ..
                } => {
                    assert_eq!(bakaze, wind);
                    assert_eq!(oya, kyoku % 3);
                    assert_eq!(k, kyoku % 3 + 1);
                }
                _ => unreachable!(),
            }
        }
    }
}
