CREATE OR REPLACE VIEW v_response_acc AS
WITH user_pool AS (
    SELECT user_id,
           bool_or(in_random) AS in_random,
           bool_or(in_top)    AS in_top
    FROM scores
    GROUP BY user_id
),
     filtered AS (
    SELECT s.score_id, s.user_id, s.beatmap_id, s.accuracy, s.score,
           u.in_random, u.in_top, b.keys,
           CASE WHEN (s.mods & (64 | 512)) <> 0 THEN 'DT'
                WHEN (s.mods & 256) <> 0        THEN 'HT'
                ELSE 'NM' END AS rate_group
    FROM scores s
    JOIN beatmaps b  ON b.beatmap_id = s.beatmap_id
    JOIN user_pool u ON u.user_id = s.user_id
    WHERE (s.mods & 17261) = s.mods
),
     best AS (
         SELECT DISTINCT ON (f.user_id, f.beatmap_id, f.rate_group)
             f.user_id, f.beatmap_id, f.rate_group, f.accuracy, f.keys,
             f.in_random, f.in_top
         FROM filtered f
         -- score_id breaks ties, so which play wins is reproducible
         ORDER BY f.user_id, f.beatmap_id, f.rate_group,
                  f.accuracy DESC, f.score DESC, f.score_id
     )
SELECT b.user_id, b.beatmap_id, b.rate_group,
       b.accuracy AS response, b.keys, b.in_random, b.in_top
FROM best b;


CREATE OR REPLACE VIEW v_response_score AS
WITH user_pool AS (
    SELECT user_id,
           bool_or(in_random) AS in_random,
           bool_or(in_top)    AS in_top
    FROM scores
    GROUP BY user_id
),
     filtered AS (
    SELECT s.score_id, s.user_id, s.beatmap_id, s.accuracy,
           u.in_random, u.in_top, b.keys,
           CASE WHEN (s.mods & (64 | 512)) <> 0 THEN 'DT'
                WHEN (s.mods & 256) <> 0        THEN 'HT'
                ELSE 'NM' END AS rate_group,
           (s.score / (
               1000000.0
               * CASE WHEN (s.mods & 1)   <> 0 THEN 0.5 ELSE 1.0 END
               * CASE WHEN (s.mods & 256) <> 0 THEN 0.5 ELSE 1.0 END
           ))::double precision AS response
    FROM scores s
    JOIN beatmaps b  ON b.beatmap_id = s.beatmap_id
    JOIN user_pool u ON u.user_id = s.user_id
    WHERE (s.mods & 17261) = s.mods
),
     best AS (
         SELECT DISTINCT ON (f.user_id, f.beatmap_id, f.rate_group)
             f.user_id, f.beatmap_id, f.rate_group, f.response, f.keys,
             f.in_random, f.in_top
         FROM filtered f
         ORDER BY f.user_id, f.beatmap_id, f.rate_group,
                  f.response DESC, f.accuracy DESC, f.score_id
     )
SELECT b.user_id, b.beatmap_id, b.rate_group, b.response, b.keys,
       b.in_random, b.in_top
FROM best b;