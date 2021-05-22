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

CREATE TABLE clean_reviews(
	key VARCHAR(15) NOT NULL,
	stars INT,
	helpful_yes FLOAT,
	helpful_no FLOAT,
	text VARCHAR(10000),
	FOREIGN KEY (key) REFERENCES products(key)
);

CREATE TABLE high_rating(
	key VARCHAR(15) NOT NULL,
	name VARCHAR(255) NOT NULL,
	description VARCHAR(10000),
	rating FLOAT,
	rating_count INT,
	FOREIGN KEY (key) REFERENCES products(key)
);

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

SELECT * FROM combined