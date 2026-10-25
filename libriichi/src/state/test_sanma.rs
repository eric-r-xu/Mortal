use super::PlayerState;
use crate::consts::LIVE_WALL;
use crate::tu8;

const UNKNOWN_HAND: &str = r#"["?","?","?","?","?","?","?","?","?","?","?","?","?"]"#;

/// A sanma state after `start_kyoku`, with `hand` (13 tiles) for `player_id`
/// and seat 0 as the dealer.
fn start(player_id: u8, hand: &str, dora_marker: &str) -> PlayerState {
    let mut tehais = [UNKNOWN_HAND; 3];
    tehais[player_id as usize] = hand;
    let mut ps = PlayerState::new(player_id);
    ps.update_json(&format!(
        r#"{{"type":"start_kyoku","bakaze":"E","dora_marker":"{dora_marker}","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[35000,35000,35000],"tehais":[{},{},{}]}}"#,
        tehais[0], tehais[1], tehais[2],
    ))
    .unwrap();
    ps
}

fn tsumo(ps: &mut PlayerState, actor: u8, pai: &str) {
    ps.update_json(&format!(
        r#"{{"type":"tsumo","actor":{actor},"pai":"{pai}"}}"#
    ))
    .unwrap();
}

fn kita(ps: &mut PlayerState, actor: u8) {
    ps.update_json(&format!(
        r#"{{"type":"nukidora","actor":{actor},"pai":"N"}}"#
    ))
    .unwrap();
}

#[test]
fn live_wall_is_55() {
    let ps = start(
        0,
        r#"["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","E"]"#,
        "1s",
    );
    assert_eq!(LIVE_WALL, 55);
    assert_eq!(ps.tiles_left, 55);
}

#[test]
fn own_kita_leaves_the_hand_and_is_a_dora() {
    let hand = r#"["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","N"]"#;
    // A Chun indicator makes Haku the dora, which the hand does not hold.
    let mut ps = start(0, hand, "C");
    tsumo(&mut ps, 0, "5s");
    assert_eq!(ps.doras_owned[0], 0);
    kita(&mut ps, 0);
    assert_eq!(ps.tehai[tu8!(N) as usize], 0);
    assert_eq!(ps.kitas, [1, 0, 0]);
    assert_eq!(ps.doras_owned[0], 1);
    assert!(ps.at_rinshan);
    assert!(!ps.can_w_riichi);

    // The replacement draw comes out of the live wall count like a rinshan.
    tsumo(&mut ps, 0, "6s");
    assert_eq!(ps.tiles_left, LIVE_WALL - 2);
}

#[test]
fn kita_counts_twice_when_north_is_dora() {
    // A West indicator makes North the dora.
    let hand = r#"["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","N","N"]"#;
    let mut ps = start(0, hand, "W");
    assert_eq!(ps.doras_owned[0], 2);
    tsumo(&mut ps, 0, "5s");
    kita(&mut ps, 0);
    // Both Norths are still ours as doras, and the kita adds one.
    assert_eq!(ps.doras_owned[0], 3);

    // An opponent's kita is worth its dora factor plus one.
    kita(&mut ps, 1);
    assert_eq!(ps.doras_owned[1], 2);
}

#[test]
fn a_later_dora_indicator_counts_existing_kitas() {
    let hand = r#"["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","E"]"#;
    let mut ps = start(0, hand, "1s");
    kita(&mut ps, 1);
    kita(&mut ps, 1);
    assert_eq!(ps.doras_owned[1], 2);
    ps.update_json(r#"{"type":"dora","dora_marker":"W"}"#)
        .unwrap();
    assert_eq!(ps.doras_owned[1], 4);
}

#[test]
fn ron_on_a_kita_then_same_cycle_furiten() {
    // 123456789p (ittsuu) 123s, waiting on a North tanki.
    let hand = r#"["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","N"]"#;
    let mut ps = start(1, hand, "1s");
    assert!(ps.waits[tu8!(N) as usize]);
    tsumo(&mut ps, 0, "?");
    kita(&mut ps, 0);
    assert!(ps.last_cans.can_ron_agari);
    assert_eq!(ps.last_kawa_tile.unwrap().as_u8(), tu8!(N));

    // Letting it go is furiten from the next event on.
    tsumo(&mut ps, 0, "?");
    assert!(ps.at_furiten);
}

#[test]
fn kita_ends_the_first_go_around() {
    // Nine kinds of terminals and honors.
    let hand = r#"["1m","9m","1p","9p","1s","9s","E","S","W","P","F","C","2p"]"#;

    let mut ps = start(1, hand, "2s");
    tsumo(&mut ps, 0, "?");
    ps.update_json(r#"{"type":"dahai","actor":0,"pai":"3s","tsumogiri":true}"#)
        .unwrap();
    tsumo(&mut ps, 1, "3p");
    assert!(ps.last_cans.can_ryukyoku);

    let mut ps = start(1, hand, "2s");
    tsumo(&mut ps, 0, "?");
    kita(&mut ps, 0);
    tsumo(&mut ps, 0, "?");
    ps.update_json(r#"{"type":"dahai","actor":0,"pai":"3s","tsumogiri":true}"#)
        .unwrap();
    tsumo(&mut ps, 1, "3p");
    assert!(!ps.can_w_riichi);
    assert!(!ps.last_cans.can_ryukyoku);
}

#[test]
fn kita_is_offered_after_an_own_draw_with_a_north() {
    let hand = r#"["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","N","N"]"#;
    let mut ps = start(0, hand, "C");
    tsumo(&mut ps, 0, "5s");
    assert!(ps.last_cans.can_kita);
    ps.validate_reaction_json(r#"{"type":"nukidora","actor":0,"pai":"N"}"#)
        .unwrap();

    // Again after the replacement draw, while a North is left.
    kita(&mut ps, 0);
    tsumo(&mut ps, 0, "6s");
    assert!(ps.last_cans.can_kita);
    kita(&mut ps, 0);
    tsumo(&mut ps, 0, "7s");
    assert!(!ps.last_cans.can_kita);
    ps.validate_reaction_json(r#"{"type":"nukidora","actor":0,"pai":"N"}"#)
        .unwrap_err();
}

#[test]
fn no_kita_after_a_pon_or_on_the_last_draw() {
    let hand = r#"["1p","1p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","N"]"#;
    let mut ps = start(1, hand, "C");
    tsumo(&mut ps, 0, "?");
    ps.update_json(r#"{"type":"dahai","actor":0,"pai":"1p","tsumogiri":true}"#)
        .unwrap();
    ps.update_json(r#"{"type":"pon","actor":1,"target":0,"pai":"1p","consumed":["1p","1p"]}"#)
        .unwrap();
    assert!(ps.last_cans.can_discard);
    assert!(!ps.last_cans.can_kita);

    let mut ps = start(1, hand, "C");
    for _ in 0..LIVE_WALL - 1 {
        tsumo(&mut ps, 0, "?");
    }
    tsumo(&mut ps, 1, "N");
    assert_eq!(ps.tiles_left, 0);
    assert!(!ps.last_cans.can_kita);
}

#[test]
fn in_riichi_only_the_drawn_north_can_be_set_aside() {
    // 123456789p 123s with a North tanki, already in riichi.
    let hand = r#"["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","N"]"#;
    let mut ps = start(0, hand, "C");
    tsumo(&mut ps, 0, "W");
    ps.update_json(r#"{"type":"reach","actor":0}"#).unwrap();
    ps.update_json(r#"{"type":"dahai","actor":0,"pai":"W","tsumogiri":true}"#)
        .unwrap();
    ps.update_json(r#"{"type":"reach_accepted","actor":0}"#)
        .unwrap();

    tsumo(&mut ps, 0, "9s");
    assert!(!ps.last_cans.can_kita, "the North in hand is the wait");
    ps.update_json(r#"{"type":"dahai","actor":0,"pai":"9s","tsumogiri":true}"#)
        .unwrap();

    // A drawn North would be a win here, so make the wait something else.
    let hand = r#"["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","E"]"#;
    let mut ps = start(0, hand, "C");
    tsumo(&mut ps, 0, "W");
    ps.update_json(r#"{"type":"reach","actor":0}"#).unwrap();
    ps.update_json(r#"{"type":"dahai","actor":0,"pai":"W","tsumogiri":true}"#)
        .unwrap();
    ps.update_json(r#"{"type":"reach_accepted","actor":0}"#)
        .unwrap();
    tsumo(&mut ps, 0, "N");
    assert!(ps.last_cans.can_kita);
}

#[test]
fn an_opponent_kita_ends_ippatsu_unless_it_is_ronned() {
    // In riichi on a North tanki, with ippatsu.
    let hand = r#"["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","N"]"#;
    let riichi = |ps: &mut PlayerState| {
        tsumo(ps, 0, "W");
        ps.update_json(r#"{"type":"reach","actor":0}"#).unwrap();
        ps.update_json(r#"{"type":"dahai","actor":0,"pai":"W","tsumogiri":true}"#)
            .unwrap();
        ps.update_json(r#"{"type":"reach_accepted","actor":0}"#)
            .unwrap();
        assert!(ps.at_ippatsu);
    };

    // A kita we can ron keeps ippatsu for that ron, then it is gone.
    let mut ps = start(0, hand, "C");
    riichi(&mut ps);
    tsumo(&mut ps, 1, "?");
    kita(&mut ps, 1);
    assert!(ps.last_cans.can_ron_agari);
    assert!(ps.at_ippatsu);
    tsumo(&mut ps, 1, "?");
    assert!(!ps.at_ippatsu);

    // Waiting on something else, the kita ends ippatsu at once.
    let hand = r#"["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","E"]"#;
    let mut ps = start(0, hand, "C");
    riichi(&mut ps);
    tsumo(&mut ps, 1, "?");
    kita(&mut ps, 1);
    assert!(!ps.last_cans.can_ron_agari);
    assert!(!ps.at_ippatsu);
}
