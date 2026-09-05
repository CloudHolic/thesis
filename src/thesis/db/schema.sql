CREATE TABLE IF NOT EXISTS scores (
    score_id    BIGINT      NOT NULL,
    user_id     INT         NOT NULL,
    beatmap_id  INT         NOT NULL,
    rank        VARCHAR(2)  NOT NULL,
    count_max   INT         NOT NULL DEFAULT 0,
    count_300   INT         NOT NULL DEFAULT 0,
    count_200   INT         NOT NULL DEFAULT 0,
    count_100   INT         NOT NULL DEFAULT 0,
    count_50    INT         NOT NULL DEFAULT 0,
    count_miss  INT         NOT NULL DEFAULT 0,
    score       INT         NOT NULL DEFAULT 0,
    accuracy    FLOAT       NOT NULL,
    mods        INT         NOT NULL DEFAULT 0,
    pp          FLOAT,
    date        TIMESTAMP   NOT NULL,
    in_top      BOOLEAN     NOT NULL DEFAULT FALSE,
    in_random   BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (score_id)
);

CREATE INDEX IF NOT EXISTS idx_scores_user_beatmap
    ON scores (user_id, beatmap_id);
CREATE INDEX IF NOT EXISTS idx_scores_beatmap
    ON scores (beatmap_id);


CREATE TABLE IF NOT EXISTS beatmaps (
    beatmap_id      INT     PRIMARY KEY,
    beatmapset_id   INT,
    keys            INT,
    star_rating     FLOAT,
    length          INT,
    play_count      INT,
    pass_count      INT,
    bpm             FLOAT,
    version         TEXT
);

CREATE TABLE IF NOT EXISTS ingest_log (
    id              BIGSERIAL   PRIMARY KEY,
    dump_file       TEXT        NOT NULL,
    member          TEXT        NOT NULL,
    target_table    TEXT        NOT NULL,
    in_top          BOOLEAN     NOT NULL,
    in_random       BOOLEAN     NOT NULL,
    rows_parsed     BIGINT      NOT NULL,
    rows_written    BIGINT      NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ NOT NULL
);