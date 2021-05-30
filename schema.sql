-- Creating ice cream tables

-- Creating products table
CREATE TABLE products(
	brand VARCHAR(15) NOT NULL,
	key VARCHAR(15) NOT NULL,
	name VARCHAR(255) NOT NULL,
	subhead VARCHAR(255),
	description VARCHAR(10000),
	rating FLOAT,
	rating_count INT,
	ingredients VARCHAR(10000),
	PRIMARY KEY (key)
);

-- Create Clean Reviews table
CREATE TABLE clean_reviews(
	key VARCHAR(15) NOT NULL,
	stars INT,
	helpful_yes FLOAT,
	helpful_no FLOAT,
	text VARCHAR(10000),
	FOREIGN KEY (key) REFERENCES products(key)
);

-- View Clean Reviews table
SELECT * FROM clean_reviews;
-- View Products table
SELECT * FROM products;

-- Create table that filters products table by rating > 4
CREATE TABLE high_rating(
	key VARCHAR(15) NOT NULL,
	name VARCHAR(255) NOT NULL,
	description VARCHAR(10000),
	rating FLOAT,
	rating_count INT,
	FOREIGN KEY (key) REFERENCES products(key)
);

-- View High Ratings table
SELECT * FROM high_rating;

-- Create table that combines High Rating and Clean Reviews via inner join
SELECT hr.key, 
	hr.name,
	hr.description,
	hr.rating,
	hr.rating_count,
	cr.stars,
	cr.helpful_yes,
	cr.helpful_no,
	cr.text
INTO combined
FROM high_rating as hr
INNER JOIN clean_reviews as cr
ON hr.key = cr.key
ORDER BY hr.key;

-- View Combined table
SELECT * FROM combined;

-- Create Helpful Clean Reviews Combined table
SELECT
	co.key,
	co.stars,
	co.helpful_yes,
	co.helpful_no,
	co.text,
	co.rating
INTO helpful_clean_reviews_combined
FROM combined as co
WHERE
	co.helpful_yes > co.helpful_no;
	
-- View Helpful Clean Reviews Combined table
SELECT * FROM helpful_clean_reviews_combined;
