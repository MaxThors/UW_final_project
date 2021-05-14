-- Creating ice cream tables
CREATE TABLE products(
	brand VARCHAR(10) NOT NULL,
	key VARCHAR(5) NOT NULL,
	name VARCHAR(40) NOT NULL,
	subhead VARCHAR(255),
	description VARCHAR(1000),
	rating FLOAT,
	rating_count INT,
	ingredients VARCHAR(10000),
	PRIMARY KEY (key)
);
	
CREATE TABLE reviews(
	brand VARCHAR(10) NOT NULL,
	key VARCHAR(5) NOT NULL,
	author VARCHAR(40),
	date VARCHAR(10),
	stars INT,
	title VARCHAR(100),
	helpful_yes INT,
	helpful_no INT,
	text VARCHAR(1000),
	taste INT,
	ingredients INT,
	texture INT,
	likes INT,
	FOREIGN KEY (key) REFERENCES products(key)
);

SELECT * FROM reviews;
SELECT * FROM products;