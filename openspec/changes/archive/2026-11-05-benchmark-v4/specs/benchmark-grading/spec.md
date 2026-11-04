## ADDED Requirements

### Requirement: Benchmark v4 grades on-lawn activity from positions

Benchmark v4 SHALL add an activity family on Scheer's small lawn to v3's 21 statistics, whose bands
and readings SHALL stay byte-identical. Over a worm's on-lawn seconds (both samples present and on the
lawn) it SHALL count:

- pauses: displacements shorter than 0.02 mm, the floor below which 95% of the real tracker's
  still-worm 1-s readings fall;
- direction flips: v3's rule (pairs at least 0.03 mm, turning more than 135°) on the on-lawn track;
- reversal onsets: the on-lawn detector at 0.03 mm.

It SHALL grade the pause share and the flip fraction, and the onset rate per on-lawn minute only if
the reachability oracle reaches it; otherwise the onset rate SHALL be reported, not graded. The bar
SHALL be z\* = 3.0781 over 24 statistics, or 3.0654 over 23. D_fit SHALL be the small lawn, the
uniform well, kinematics and activity. Every activity statistic SHALL be read the same way on real
tracks, on simulated tracks after Scheer's rig model, and on imported tracks. v1, v2 and v3 SHALL be
unchanged.

#### Scenario: A worm that does not run forward fails

- **WHEN** a population's on-lawn motion is reversals from standstill, with no forward runs
- **THEN** its on-lawn flip fraction z SHALL exceed the bar

#### Scenario: A worm that never pauses fails

- **WHEN** a population's worms move on the lawn without pausing, as the constant-hazard negative does
- **THEN** its pause-share z SHALL exceed the bar

#### Scenario: v3 is unchanged

- **WHEN** any population is graded on v3 after v4 is added
- **THEN** every v3 statistic, D and presence reading SHALL be byte-identical to before

#### Scenario: Imported tracks carry the activity family

- **WHEN** real tracks are exported and imported through CSV or WCON
- **THEN** their activity tallies SHALL equal the native path's
