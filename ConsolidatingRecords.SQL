UPDATE records SET education_level_id = (SELECT name FROM education_levels WHERE records.education_level_id = education_levels.id);
UPDATE records SET workclass_id = (SELECT name FROM workclasses WHERE records.workclass_id = workclasses.id);
UPDATE records SET marital_status_id = (SELECT name FROM marital_statuses WHERE records.marital_status_id = marital_statuses.id);
UPDATE records SET occupation_id = (SELECT name FROM occupations WHERE records.occupation_id = occupations.id);
UPDATE records SET country_id = (SELECT name FROM countries WHERE records.country_id = countries.id);
UPDATE records SET relationship_id = (SELECT name FROM relationships WHERE records.relationship_id = relationships.id);
UPDATE records SET race_id = (SELECT name FROM races WHERE records.race_id = races.id);
UPDATE records SET sex_id = (SELECT name FROM sexes WHERE records.sex_id = sexes.id);
