-- -- SELECT DISTINCT po.procedure_concept_id, c.concept_name
-- -- FROM `sccm-discovery.rediscover_datathon_2025.procedure_occurrence` po
-- -- JOIN `sccm-discovery.rediscover_datathon_2025.concept` c
-- --   ON po.procedure_concept_id = c.concept_id
-- -- WHERE LOWER(c.concept_name) LIKE '%consult%'
-- -- LIMIT 200;


-- -- SELECT COUNT(*) as total_count
-- -- FROM `sccm-discovery.rediscover_datathon_2025.care_site`
-- -- LIMIT 100

-- -- Device exposure - get ventilator
-- -- SELECT DISTINCT de.device_concept_id, c.concept_name
-- -- FROM `sccm-discovery.rediscover_datathon_2025.device_exposure` de
-- -- JOIN `sccm-discovery.rediscover_datathon_2025.concept` c
-- --   ON de.device_concept_id = c.concept_id
-- -- WHERE LOWER(c.concept_name) LIKE '%ventilator%'

-- -- Device exposure - o2 supply

-- -- SELECT DISTINCT de.device_concept_id, c.concept_name
-- -- FROM `sccm-discovery.rediscover_datathon_2025.device_exposure` de
-- -- JOIN `sccm-discovery.rediscover_datathon_2025.concept` c
-- --   ON de.device_concept_id = c.concept_id
-- -- WHERE LOWER(c.concept_name) LIKE '%oxygen%'

-- -- Analyse LOS from visit_occurrence

-- -- SELECT 
-- --   visit_occurrence_id,
-- --   person_id,
-- --   visit_start_date,
-- --   visit_end_date,
-- --   date_diff(visit_end_date, visit_start_date, day) as los
-- -- FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence`


-- -- Mortality Analysis
-- -- See when they died, how long after their LATEST discharge
-- -- Only considers the most recent visit for each person
-- -- Filters for patients aged >= 18 and removes those who died before their visit
-- -- Ensures each person_id appears only once by handling potential duplicate death records

-- -- WITH latest_visits AS (
-- --   SELECT 
-- --     vo.person_id,
-- --     vo.visit_occurrence_id,
-- --     vo.visit_start_datetime,
-- --     vo.visit_end_datetime,
-- --     ROW_NUMBER() OVER (PARTITION BY vo.person_id ORDER BY vo.visit_end_datetime DESC) as rn
-- --   FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence` vo
-- --   JOIN `sccm-discovery.rediscover_datathon_2025.person` p
-- --     ON vo.person_id = p.person_id
-- --   WHERE EXTRACT(YEAR FROM vo.visit_start_datetime) - p.year_of_birth >= 18  -- Age >= 18 at visit
-- -- ),
-- -- latest_deaths AS (
-- --   SELECT 
-- --     person_id,
-- --     death_datetime,
-- --     ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY death_datetime DESC) as death_rn
-- --   FROM `sccm-discovery.rediscover_datathon_2025.death`
-- --   WHERE death_datetime IS NOT NULL
-- -- )
-- -- SELECT 
-- --   lv.person_id,
-- --   lv.visit_occurrence_id,
-- --   lv.visit_start_datetime,
-- --   lv.visit_end_datetime as latest_discharge_datetime,
-- --   ld.death_datetime,
-- --   datetime_diff(ld.death_datetime, lv.visit_end_datetime, day) as days_to_death,
-- --   CASE 
-- --     WHEN ld.death_datetime >= lv.visit_end_datetime THEN 'Death after discharge'
-- --     WHEN ld.death_datetime BETWEEN lv.visit_start_datetime AND lv.visit_end_datetime THEN 'Death during visit'
-- --   END as death_timing
-- -- FROM latest_visits lv
-- -- JOIN latest_deaths ld
-- --   ON lv.person_id = ld.person_id
-- -- WHERE lv.rn = 1  -- Only the latest visit per person
-- --   AND ld.death_rn = 1  -- Only the latest death record per person
-- --   -- AND ld.death_datetime >= lv.visit_end_datetime  -- Remove patients who died before their visit
-- -- ORDER BY days_to_death DESC


-- -- SELECT visit_type_concept_id FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence` a INNER JOIN `sccm-discovery.rediscover_datathon_2025.concept` b by a.visit_type_concept_id

-- -- Optimized query to get unique visit type concept names
-- -- SELECT DISTINCT b.concept_name 
-- -- FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence` a 
-- -- INNER JOIN `sccm-discovery.rediscover_datathon_2025.concept` b 
-- --   ON a.visit_concept_id = b.concept_id
-- -- WHERE b.concept_name IS NOT NULL
-- -- ORDER BY b.concept_name

-- -- SELECT DISTINCT visit_concept_id FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence`





-- -- SELECT p.race_concept_id, c.concept_name, COUNT(*) as count 
-- -- FROM `sccm-discovery.rediscover_datathon_2025.person` p
-- -- INNER JOIN `sccm-discovery.rediscover_datathon_2025.concept` c 
-- --   ON p.race_concept_id = c.concept_id 
-- -- WHERE c.concept_name IS NOT NULL
-- -- GROUP BY p.race_concept_id, c.concept_name


-- -- Count total visit occurrences and distinct persons where they are 18 or older
-- -- SELECT 
-- --   DISTINCT vo.person_id
-- -- FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence` vo
-- -- JOIN `sccm-discovery.rediscover_datathon_2025.person` p
-- --   ON vo.person_id = p.person_id
-- -- WHERE EXTRACT(YEAR FROM vo.visit_start_datetime) - p.year_of_birth >= 18

-- -- SELECT COUNT(*) as count 
-- -- FROM `sccm-discovery.rediscover_datathon_2025.person`

-- SELECT co.condition_status_concept_id, c.concept_name, COUNT(*) as count
-- FROM `sccm-discovery.rediscover_datathon_2025.condition_occurrence` co
-- JOIN `sccm-discovery.rediscover_datathon_2025.concept` c
--   ON co.condition_status_concept_id = c.concept_id
-- GROUP BY co.condition_status_concept_id, c.concept_name
-- ORDER BY count DESC

/* ------------------------------------------------------------------ */
/*  1.  Seed concept IDs for each drug class                          */
/* ------------------------------------------------------------------ */
WITH drug_class_seeds AS (
  -- Use UNION ALL so every concept keeps its class label
  SELECT 'steroid'    AS drug_class, cid AS concept_id
    FROM UNNEST([36681928, 4228567, 36682298]) cid
  UNION ALL
  SELECT 'narcotic',  cid FROM UNNEST([36681676, 36681935]) cid
  UNION ALL
  SELECT 'sedative',  cid FROM UNNEST([36682329, 4252075])  cid
  UNION ALL
  SELECT 'vasopressor', cid
    FROM UNNEST([4276507, 4183404, 4009018, 4322956, 4132146]) cid
)

/* ------------------------------------------------------------------ */
/*  2.  Find every visit in which that drug class was given           */
/*      (same join chain you already use for steroids)                */
/* ------------------------------------------------------------------ */
, drug_class_visits AS (
  SELECT DISTINCT
         i.person_id,
         i.visit_occurrence_id,
         s.drug_class
  FROM  drug_class_seeds                                            AS s
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept`           AS a
        ON a.concept_id = s.concept_id
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept_relationship` AS b
        ON  b.concept_id_1   = a.concept_id
       AND b.relationship_id = 'Subsumes'
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept`           AS c
        ON  c.concept_id     = b.concept_id_2
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept_relationship` AS d
        ON  d.concept_id_1   = c.concept_id
       AND d.relationship_id = 'SNOMED - RxNorm eq'
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept`           AS e
        ON e.concept_id      = d.concept_id_2
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept_relationship` AS f
        ON f.concept_id_1    = e.concept_id
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept`           AS g
        ON  g.concept_id      = f.concept_id_2
       AND g.standard_concept = 'S'       -- keep standard drug concepts only
  JOIN  `sccm-discovery.rediscover_datathon_2025.drug_exposure`     AS h
        ON  h.drug_concept_id = g.concept_id
  JOIN  `sccm-discovery.rediscover_datathon_2025.visit_occurrence`  AS i
        ON  i.visit_occurrence_id = h.visit_occurrence_id
  JOIN  `sccm-discovery.rediscover_datathon_2025.person`            AS p
        ON  p.person_id = i.person_id
  WHERE EXTRACT(YEAR FROM i.visit_start_date) - p.year_of_birth >= 18   -- ≥ 18 y old
)

/* ------------------------------------------------------------------ */
/*  3.  Pivot those rows into one record per (person, visit) with     */
/*      four 0/1 flags                                                */
/* ------------------------------------------------------------------ */
, drug_class_flags AS (
  SELECT
    person_id,
    visit_occurrence_id,
    MAX(CASE WHEN drug_class = 'steroid'     THEN 1 END) AS steroid_flag,
    MAX(CASE WHEN drug_class = 'narcotic'    THEN 1 END) AS narcotic_flag,
    MAX(CASE WHEN drug_class = 'sedative'    THEN 1 END) AS sedative_flag,
    MAX(CASE WHEN drug_class = 'vasopressor' THEN 1 END) AS vasopressor_flag
  FROM  drug_class_visits
  GROUP BY person_id, visit_occurrence_id
)

/* ------------------------------------------------------------------ */
/*  4.  Attach the flags to your cohort table                         */
/* ------------------------------------------------------------------ */
SELECT
  c.*,
  IFNULL(f.steroid_flag,     0) AS steroid_flag,
  IFNULL(f.narcotic_flag,    0) AS narcotic_flag,
  IFNULL(f.sedative_flag,    0) AS sedative_flag,
  IFNULL(f.vasopressor_flag, 0) AS vasopressor_flag
FROM  `sccm-discovery.rediscover_datathon_2025.cohort_team_2`  AS c
LEFT JOIN drug_class_flags                                       AS f
       ON  f.person_id          = c.person_id
      AND f.visit_occurrence_id = c.visit_occurrence_id ;
