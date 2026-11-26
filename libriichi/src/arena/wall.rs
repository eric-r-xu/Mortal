//! The unshuffled walls the arena boards deal from.

use crate::t;
use crate::tile::Tile;

#[rustfmt::skip]
#[cfg_attr(feature = "sanma", allow(dead_code))] // the sanma board deals `SANMA_UNSHUFFLED`
pub(super) const UNSHUFFLED: [Tile; 136] = [
    t!(1m),  t!(1m), t!(1m), t!(1m),
    t!(2m),  t!(2m), t!(2m), t!(2m),
    t!(3m),  t!(3m), t!(3m), t!(3m),
    t!(4m),  t!(4m), t!(4m), t!(4m),
    t!(5mr), t!(5m), t!(5m), t!(5m),
    t!(6m),  t!(6m), t!(6m), t!(6m),
    t!(7m),  t!(7m), t!(7m), t!(7m),
    t!(8m),  t!(8m), t!(8m), t!(8m),
    t!(9m),  t!(9m), t!(9m), t!(9m),

    t!(1p),  t!(1p), t!(1p), t!(1p),
    t!(2p),  t!(2p), t!(2p), t!(2p),
    t!(3p),  t!(3p), t!(3p), t!(3p),
    t!(4p),  t!(4p), t!(4p), t!(4p),
    t!(5pr), t!(5p), t!(5p), t!(5p),
    t!(6p),  t!(6p), t!(6p), t!(6p),
    t!(7p),  t!(7p), t!(7p), t!(7p),
    t!(8p),  t!(8p), t!(8p), t!(8p),
    t!(9p),  t!(9p), t!(9p), t!(9p),

    t!(1s),  t!(1s), t!(1s), t!(1s),
    t!(2s),  t!(2s), t!(2s), t!(2s),
    t!(3s),  t!(3s), t!(3s), t!(3s),
    t!(4s),  t!(4s), t!(4s), t!(4s),
    t!(5sr), t!(5s), t!(5s), t!(5s),
    t!(6s),  t!(6s), t!(6s), t!(6s),
    t!(7s),  t!(7s), t!(7s), t!(7s),
    t!(8s),  t!(8s), t!(8s), t!(8s),
    t!(9s),  t!(9s), t!(9s), t!(9s),

    t!(E), t!(E), t!(E), t!(E),
    t!(S), t!(S), t!(S), t!(S),
    t!(W), t!(W), t!(W), t!(W),
    t!(N), t!(N), t!(N), t!(N),
    t!(P), t!(P), t!(P), t!(P),
    t!(F), t!(F), t!(F), t!(F),
    t!(C), t!(C), t!(C), t!(C),
];

/// The sanma wall: `UNSHUFFLED` without 2m-8m.
// One red 5p and one red 5s, as in Tenhou phoenix-room sanma logs (the aka-dora counts of 2,496 winning
// hands match exactly).
#[cfg_attr(not(feature = "sanma"), allow(dead_code))] // dealt only by the sanma board
pub(super) const SANMA_UNSHUFFLED: [Tile; 108] = {
    let mut wall = [t!(?); 108];
    let mut i = 0;
    let mut n = 0;
    while i < UNSHUFFLED.len() {
        if !UNSHUFFLED[i].is_sanma_removed() {
            wall[n] = UNSHUFFLED[i];
            n += 1;
        }
        i += 1;
    }
    assert!(n == 108);
    wall
};

#[cfg(test)]
mod sanma_test {
    use super::*;

    #[test]
    fn sanma_wall_has_no_manzu_middle() {
        assert_eq!(
            SANMA_UNSHUFFLED
                .iter()
                .filter(|t| t.deaka() == t!(1m))
                .count(),
            4
        );
        assert_eq!(
            SANMA_UNSHUFFLED
                .iter()
                .filter(|t| t.deaka() == t!(9m))
                .count(),
            4
        );
        assert!(SANMA_UNSHUFFLED.iter().all(|t| !t.is_sanma_removed()));
        assert_eq!(SANMA_UNSHUFFLED.iter().filter(|t| t.is_aka()).count(), 2);
    }
}
