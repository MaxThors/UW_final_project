-- Creating ice cream tables
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
	
CREATE TABLE reviews(
	key VARCHAR(15) NOT NULL,
	stars INT,
	helpful_yes FLOAT,
	helpful_no FLOAT,
	text VARCHAR(10000),
	FOREIGN KEY (key) REFERENCES products(key)
);

SELECT * FROM reviews;
SELECT * FROM products;