-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE "products" (
    "brand" varchar   NOT NULL,
    "key" varchar   NOT NULL,
    "name" varchar   NOT NULL,
    "subhead" varchar   NOT NULL,
    "description" varchar   NOT NULL,
    "rating" float   NOT NULL,
    "rating_count" int   NOT NULL,
    "ingredients" varchar   NOT NULL,
    CONSTRAINT "pk_products" PRIMARY KEY (
        "key"
     )
);

CREATE TABLE "reviews" (
    "brand" varchar   NOT NULL,
    "key" varchar   NOT NULL,
    "author" varchar   NOT NULL,
    "date" datetime   NOT NULL,
    "stars" int   NOT NULL,
    "title" varchar   NOT NULL,
    "helpful_yes" int   NOT NULL,
    "helpful_no" int   NOT NULL,
    "text" varchar   NOT NULL,
    "taste" int   NOT NULL,
    "ingredients" int   NOT NULL,
    "texture" int   NOT NULL,
    "likes" int   NOT NULL
);

ALTER TABLE "reviews" ADD CONSTRAINT "fk_reviews_key" FOREIGN KEY("key")
REFERENCES "products" ("key");

