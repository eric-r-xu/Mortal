use crate::py_helper::add_submodule;

use pyo3::prelude::*;

/// Seats at the table: 4, or 3 when built with the `sanma` feature.
pub const NUM_PLAYERS: usize = if cfg!(feature = "sanma") { 3 } else { 4 };

pub const NUM_PLAYERS_U8: u8 = NUM_PLAYERS as u8;

/// Starting points. Sanma is 35000-point start.
pub const INIT_SCORE: i32 = if cfg!(feature = "sanma") {
    35000
} else {
    25000
};
/// The sudden-death and agari-yame target. Sanma is 40000-point return, checked against Tenhou
/// phoenix-room logs.
pub const END_SCORE: i32 = if cfg!(feature = "sanma") {
    40000
} else {
    30000
};
/// The points that change hands when a hand ends in an exhaustive draw: split among the tenpai
/// players and paid by the others. Sanma pays 2000, checked against Tenhou phoenix-room logs.
pub const NOTEN_POT: i32 = if cfg!(feature = "sanma") { 2000 } else { 3000 };
/// Draws a hand allows from the live wall: 70, or 55 in sanma (108 tiles, less the 14-tile dead
/// wall and the 39 dealt). Every exhaustive draw in the Tenhou phoenix-room sanma logs had 55.
pub const LIVE_WALL: u8 = if cfg!(feature = "sanma") { 55 } else { 70 };
/// Scheduled hands in a tonpuu game: one per seat.
pub const TONPUU: u8 = NUM_PLAYERS_U8;
/// Scheduled hands in a hanchan game: two winds.
pub const HANCHAN: u8 = 2 * NUM_PLAYERS_U8;

pub const MAX_VERSION: u32 = 4;

pub const ACTION_SPACE: usize = 37 // discard | kan (choice)
                              + 1  // riichi
                              + 3  // chi
                              + 1  // pon
                              + 1  // kan (decide)
                              + 1  // agari
                              + 1  // ryukyoku
                              + 1  // pass
                              + if cfg!(feature = "sanma") { 1 } else { 0 }; // kita
// = 46, or 47 in sanma
// Grand kyoku, honba and kyotaku, then one score per seat.
pub const GRP_SIZE: usize = 3 + NUM_PLAYERS;

#[pyfunction]
#[inline]
pub const fn obs_shape(version: u32) -> (usize, usize) {
    match version {
        1 => (938, 34),
        2 => (942, 34),
        3 => (934, 34),
        4 => (1012, 34),
        _ => unreachable!(),
    }
}

#[pyfunction]
#[inline]
pub const fn oracle_obs_shape(version: u32) -> (usize, usize) {
    match version {
        1 => (211, 34),
        2 | 3 | 4 => (217, 34),
        _ => unreachable!(),
    }
}

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "consts")?;
    m.add_function(wrap_pyfunction!(obs_shape, &m)?)?;
    m.add_function(wrap_pyfunction!(oracle_obs_shape, &m)?)?;
    m.add("MAX_VERSION", MAX_VERSION)?;
    m.add("ACTION_SPACE", ACTION_SPACE)?;
    m.add("GRP_SIZE", GRP_SIZE)?;
    add_submodule(py, prefix, super_mod, &m)
}
