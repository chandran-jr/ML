CREATE TABLE crimedata (
    crimeID int(4),
    FirstName varchar(255),
    LastName varchar(255),
    City varchar(255),
    Hours int,
    Minutes int,
    Crime varchar(255),
    latitude float(9,6),
    longitude float(9,6)
);

INSERT INTO crimedata (crimeID, FirstName, LastName, City, Hours, Minutes, Crime, latitude, longitude)
VALUES (001, 'Paulinho', 'Joseph', 'London', 22, 36, 'Murder', 35.123456, 123.456312);

INSERT INTO crimedata (crimeID, FirstName, LastName, City, Hours, Minutes, Crime, latitude, longitude)
VALUES (002, 'Chris', 'Jordie', 'Paris', 01, 40, 'Theft', 120.123456, 150.456312);

INSERT INTO crimedata (crimeID, FirstName, LastName, City, Hours, Minutes, Crime, latitude, longitude)
VALUES (003, 'Mark', 'Stagen', 'Spain', 06, 22, 'Theft', 30.123456, 80.456312);

INSERT INTO crimedata (crimeID, FirstName, LastName, City, Hours, Minutes, Crime, latitude, longitude)
VALUES (004, 'Daniel', 'Joseph', 'London', 10, 40, 'Vehicle collision', 31.133456, 100.456312);

INSERT INTO crimedata (crimeID, FirstName, LastName, City, Hours, Minutes, Crime, latitude, longitude)
VALUES (005, 'Jacob', 'Thomas', 'Alapuzha', 00, 36, 'Murder', 100.123456, 123.456312);

INSERT INTO crimedata (crimeID, FirstName, LastName, City, Hours, Minutes, Crime, latitude, longitude)
VALUES (006, 'Chris', 'Henry', 'London', 20, 36, 'Vehicle collision', 119.123456, 140.456312);

INSERT INTO crimedata (crimeID, FirstName, LastName, City, Hours, Minutes, Crime, latitude, longitude)
VALUES (007, 'Joseph', 'Abraham', 'London', 11, 36, 'Fight', 35.123456, 123.456312);

INSERT INTO crimedata (crimeID, FirstName, LastName, City, Hours, Minutes, Crime, latitude, longitude)
VALUES (008, 'Sean', 'Paul', 'Vancouver', 10, 12, 'Theft', 100.123456, 210.456312);

INSERT INTO crimedata (crimeID, FirstName, LastName, City, Hours, Minutes, Crime, latitude, longitude)
VALUES (009, 'Christina', 'Thomas', 'Paris', 21, 17, 'Murder', 10.123456, 100.456312);

INSERT INTO crimedata (crimeID, FirstName, LastName, City, Hours, Minutes, Crime, latitude, longitude)
VALUES (010, 'Mindy', 'Abraham', 'Johannesberg', 11, 18, 'Vehicle Collision', 120.123456, 76.456312);


Select * FROM crimedata;
