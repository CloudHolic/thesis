CREATE OR REPLACE VIEW v_response_acc AS
WITH filtered AS (
    SELECT s.user_id, s.beatmap_id, s.accuracy, s.score, s.in_top, s.in_random,
           b.keys AS mania_keys,
           CASE WHEN (s.mods & (64 | 512)) <> 0 THEN 'DT'
                WHEN (s.mods & 256) <> 0        THEN 'HT'
                ELSE 'NM' END AS rate_group
    FROM scores s
    JOIN beatmaps b ON b.beatmap_id = s.beatmap_id
    WHERE (s.mods & 17261) = s.mods
),
     best AS (
         SELECT DISTINCT ON (f.user_id, f.beatmap_id, f.rate_group)
             f.user_id, f.beatmap_id, f.rate_group, f.accuracy, f.mania_keys,
             f.in_top, f.in_random
         FROM filtered f
         ORDER BY f.user_id, f.beatmap_id, f.rate_group, f.accuracy DESC, f.score DESC
     )
SELECT
    b.user_id, b.beatmap_id, b.rate_group,
    b.accuracy AS response,
    b.mania_keys,
    b.in_top, b.in_random
FROM best b;


CREATE OR REPLACE VIEW v_response_score AS
WITH filtered AS (
    SELECT s.user_id, s.beatmap_id, s.accuracy, s.in_top, s.in_random,
           b.keys AS mania_keys,
           CASE WHEN (s.mods & (64 | 512)) <> 0 THEN 'DT'
                WHEN (s.mods & 256) <> 0        THEN 'HT'
                ELSE 'NM' END AS rate_group,
           (s.score / (
               1000000.0
               * CASE WHEN (s.mods & 1)   <> 0 THEN 0.5 ELSE 1.0 END
               * CASE WHEN (s.mods & 256) <> 0 THEN 0.5 ELSE 1.0 END
           ))::double precision AS response
    FROM scores s
    JOIN beatmaps b ON b.beatmap_id = s.beatmap_id
    WHERE (s.mods & 17261) = s.mods
),
     best AS (
         SELECT DISTINCT ON (f.user_id, f.beatmap_id, f.rate_group)
             f.user_id, f.beatmap_id, f.rate_group, f.response, f.mania_keys,
             f.in_top, f.in_random
         FROM filtered f
         ORDER BY f.user_id, f.beatmap_id, f.rate_group, f.response DESC, f.accuracy DESC
     )
SELECT
    b.user_id, b.beatmap_id, b.rate_group,
    b.response,
    b.mania_keys,
    b.in_top, b.in_random
FROM best b;