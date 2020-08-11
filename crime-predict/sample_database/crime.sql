

CREATE TABLE crimedata (
    crimeID int Primary Key,
    name varchar(255),
    address varchar(255),
    fir_no int,
    height int,
    weight int,
    age int,
    gender varchar,
    phone int
    
);

INSERT INTO crimedata (crimeID, name, address, fir_no, height, weight, age, gender, phone)
VALUES (001, 'Paul Joseph', 'London', 100, 170, 65, 24, 'M', 9753512345);

INSERT INTO crimedata (crimeID, name, address, fir_no, height, weight, age, gender, phone)
VALUES (002, 'Chris Jordie', 'Paris', 200, 180, 87, 25,'M', 985647831);

INSERT INTO crimedata (crimeID, name, address, fir_no, height, weight, age, gender, phone)
VALUES (003, 'Mark Stagen', 'Spain', 300, 165, 64, 25,'M', 985654834);

INSERT INTO crimedata (crimeID, name, address, fir_no, height, weight, age, gender, phone)
VALUES (004, 'Daniel Joseph', 'Congo', 400, 180, 68, 30,'M', 962147831);

INSERT INTO crimedata (crimeID, name, address, fir_no, height, weight, age, gender, phone)
VALUES (005, 'Jacob Thomas', 'South Africa', 500, 175, 70, 40,'M', 988531931);

INSERT INTO crimedata (crimeID, name, address, fir_no, height, weight, age, gender, phone)
VALUES (006, 'Chris Henry', 'Moscow', 600, 189, 80, 36,'M', 989975031);

INSERT INTO crimedata (crimeID, name, address, fir_no, height, weight, age, gender, phone)
VALUES (007, 'Joseph Abraham', 'Berlin', 700, 177, 78, 48,'M', 985688909);

INSERT INTO crimedata (crimeID, name, address, fir_no, height, weight, age, gender, phone)
VALUES (008, 'Sean Paul', 'Helsinki', 800, 180, 78, 43,'M', 988876431);;

INSERT INTO crimedata (crimeID, name, address, fir_no, height, weight, age, gender, phone)
VALUES (009, 'Christina Thomas', 'Nairobi', 900, 154, 53, 30,'F', 985667751);

INSERT INTO crimedata (crimeID, name, address, fir_no, height, weight, age, gender, phone)
VALUES (010, 'Mindy Abraham', 'Rio', 950,  166, 68, 29,'F', 988315671);

Select * FROM crimedata;





CREATE TABLE Properties (
    surveynum int,
    place varchar(255),
    ownername varchar(255),
    sqft float(9,4),
    acre float,
    crimeID int Primary Key
);

INSERT INTO Properties (surveynum, place, ownername, sqft, acre, crimeID)
VALUES (1,  'London', 'Paul Abraham',  30, 0.5, 001);

INSERT INTO Properties (surveynum, place, ownername, sqft, acre, crimeID)
VALUES (2,  'Paris', 'Chris Jordie',  28, 0.5, 002);

INSERT INTO Properties (surveynum, place, ownername, sqft, acre, crimeID)
VALUES (3,  'Spain', 'Mark Stagen',  45, 0.5, 003);

INSERT INTO Properties (surveynum, place, ownername, sqft, acre, crimeID)
VALUES (4,  'Congo', 'Daniel Joseph',  37, 0.5, 004);

INSERT INTO Properties (surveynum, place, ownername, sqft, acre, crimeID)
VALUES (5,  'South Africa', 'Jacob Thomas',  33, 0.5, 005);

INSERT INTO Properties (surveynum, place, ownername, sqft, acre, crimeID)
VALUES (6,  'Moscow', 'Chris Henry',  22, 0.5, 006);

INSERT INTO Properties (surveynum, place, ownername, sqft, acre, crimeID)
VALUES (7,  'Berlin', 'Joseph Abraham',  13, 0.5, 007);

INSERT INTO Properties (surveynum, place, ownername, sqft, acre, crimeID)
VALUES (8,  'Helsinki', 'Sean Paul',  45, 0.5, 008);

INSERT INTO Properties (surveynum, place, ownername, sqft, acre, crimeID)
VALUES (9,  'Nairobi', 'Christina Thomas',  56, 0.5, 009);

INSERT INTO Properties (surveynum, place, ownername, sqft, acre, crimeID)
VALUES (10,  'Rio', 'Mindy Abraham',  34, 0.5, 010);

Select * FROM Properties;





CREATE TABLE Cases (
    witness varchar,
    actiontaken varchar,
    crimetype varchar,
    fir_no int Primary Key
);

INSERT INTO Cases (witness, actiontaken, crimetype, fir_no)
VALUES ('none', 'Prison', 'Murder', 100);

INSERT INTO Cases (witness, actiontaken, crimetype, fir_no)
VALUES ('Jensen Clay', 'Jail', 'Theft', 200);

INSERT INTO Cases (witness, actiontaken, crimetype, fir_no)
VALUES ('Thomas Henry', 'Fined', 'Vehicle collision', 300);

INSERT INTO Cases (witness, actiontaken, crimetype, fir_no)
VALUES ('Harley Thompson', 'Prison', 'Murder', 400);

INSERT INTO Cases (witness, actiontaken, crimetype, fir_no)
VALUES ('none', 'Jail', 'Theft', 500);

INSERT INTO Cases (witness, actiontaken, crimetype, fir_no)
VALUES ('none', 'Fined', 'Weed', 600);

INSERT INTO Cases (witness, actiontaken, crimetype, fir_no)
VALUES ('Jefferey Thomas', 'Fined', 'Drunk driving', 700);

INSERT INTO Cases (witness, actiontaken, crimetype, fir_no)
VALUES ('David George ', 'Jail', 'Theft', 800);

INSERT INTO Cases (witness, actiontaken, crimetype, fir_no)
VALUES ('none', 'Fined', 'Behicle collision', 900);

INSERT INTO Cases (witness, actiontaken, crimetype, fir_no)
VALUES ('Alice Brunt', 'Prison', 'Murder', 950);

Select * from Cases where crimetype='Murder';



CREATE TABLE Vehicle (
        vehicle_no varchar,
        vehicle_owner varchar,
        owner_address varchar,
        model varchar,
        colour varchar,
        crime_id varchar Primary Key
     
);
 
INSERT INTO Vehicle (vehicle_no, vehicle_owner, owner_address, model,colour, crime_id)
VALUES ('AB123', 'Paul Joseph', 'London','Tesla','grey', 001);

INSERT INTO Vehicle (vehicle_no, vehicle_owner, owner_address, model,colour, crime_id)
VALUES ('CV356', 'Chris Jordie', 'Paris','Nissan','blue', 002);

INSERT INTO Vehicle (vehicle_no, vehicle_owner, owner_address, model,colour, crime_id)
VALUES ('RET345', 'Mark Stagen', 'Spain','Nissan','white', 003);

INSERT INTO Vehicle (vehicle_no, vehicle_owner, owner_address, model,colour, crime_id)
VALUES ('UIT654', 'Daniel Joseph', 'Congo','Skoda','red', 004);

INSERT INTO Vehicle (vehicle_no, vehicle_owner, owner_address, model,colour, crime_id)
VALUES ('HUI879', 'Jacob Thomas', 'South Africa','Violkswagon','silver', 005);

INSERT INTO Vehicle (vehicle_no, vehicle_owner, owner_address, model,colour, crime_id)
VALUES ('FGT543', 'Chris Henry', 'Moscow','Audi','blue', 006);

INSERT INTO Vehicle (vehicle_no, vehicle_owner, owner_address, model,colour, crime_id)
VALUES ('YTR564', 'Joseph Abraham', 'Berlin','Nissan','black', 007);

INSERT INTO Vehicle (vehicle_no, vehicle_owner, owner_address, model,colour, crime_id)
VALUES ('UIY#&*', 'Sean Paul', 'Helsinki','BMW','yellow', 008);

INSERT INTO Vehicle (vehicle_no, vehicle_owner, owner_address, model,colour, crime_id)
VALUES ('OIP907', 'Christina Thomas', 'Nairobi','Skoda','red', 009);

INSERT INTO Vehicle (vehicle_no, vehicle_owner, owner_address, model,colour, crime_id)
VALUES ('TRE321', 'Mindy Abraham', 'Rio','Tesla','green', 010);

Select * from Vehicle
















