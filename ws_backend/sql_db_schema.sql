-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE "Characters" (
    "emp_no" int   NOT NULL,
    "emp_name" varchar   NOT NULL,
    CONSTRAINT "pk_Characters" PRIMARY KEY (
        "emp_name"
     )
);

CREATE TABLE "Episode" (
    "episode_id" int   NOT NULL,
    "title" varchar   NOT NULL,
    "season_no" int   NOT NULL,
    "episode_no" int   NOT NULL,
    CONSTRAINT "pk_Episode" PRIMARY KEY (
        "episode_id"
     )
);

CREATE TABLE "Script" (
    "episode_id" int   NOT NULL,
    "emp_name" varchar   NOT NULL,
    "line" varchar   NOT NULL,
    "line_id" int   NOT NULL,
    CONSTRAINT "pk_Script" PRIMARY KEY (
        "line_id"
     )
);

CREATE TABLE "Scene" (
    "scene_id" int   NOT NULL,
    "emp_name" varchar   NOT NULL,
    CONSTRAINT "pk_Scene" PRIMARY KEY (
        "scene_id"
     )
);

ALTER TABLE "Characters" ADD CONSTRAINT "fk_Characters_emp_name" FOREIGN KEY("emp_name")
REFERENCES "Scene" ("emp_name");

ALTER TABLE "Script" ADD CONSTRAINT "fk_Script_episode_id" FOREIGN KEY("episode_id")
REFERENCES "Episode" ("episode_id");

ALTER TABLE "Script" ADD CONSTRAINT "fk_Script_emp_name" FOREIGN KEY("emp_name")
REFERENCES "Characters" ("emp_name");

ALTER TABLE "Scene" ADD CONSTRAINT "fk_Scene_scene_id" FOREIGN KEY("scene_id")
REFERENCES "Script" ("scene_id");

