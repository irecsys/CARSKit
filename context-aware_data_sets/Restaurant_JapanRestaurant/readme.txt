userNN.train

This is train data set whose format is used for LIBSVM.
You can refer to http://www.csie.ntu.edu.tw/~cjlin/libsvm/ about the LIBSVM.

The left most value denotes class label.
 - +1 denotes positive data (with ratings of 6 to 10)
 - -1 denotes negative data (with ratings of 1 to 5)

Dimensions 1 to 17 denote restaurant parameters.
The parameters correspond to Table 1 in our paper entitled as
"Context-Aware SVM for context-dependent information recommendation."
Each value was given by binary data {0, 1}.

is equipped with ...
1: Car Parking
2: Non-smoking Section
3: Single Room
4: Karaoke
5: Live Concerts

has services of ...
6: Lunch
7: Takeout
8: All You Can Eat
9: Coupon
10: Open Late at Night

recommended for ...
11: Business Receptions
12: Banquets
13: Parties
14: Dates
15: Family

environment includes ...
16: Night View
17: Ocean View


Dimensions 18 to 31 denote context parameters.
The parameters correspond to Table 2 in the paper.
Each value was normalized to [0, 1]

Time
18: Month {1=Jan. to 12=Dec.}
19: Week {0=Monday to 6=Sunday}
20: Hour {0 to 23}

Schedule
21: Area Type {0=None, 1=Entertainment District, 3=Near Station, 4=Tourist Resort}
22: Budget (Yen) {0 to 10000}
23: Holiday {0=None, 1=A Day OFF, 2=Recess, 3=Before Holiday}

Partner
24: Num. of Male {0 to 10}
25: Num. of Female {0 to 10}
26: Lowest Age {0 to 100}
27: Highest Age {0 to 100}
28: Relation {0=None, 1=Family, 2=Boy/Girlfriend, 3=Friend, 4=Boss, 5=Subordinate}
29: Status {0=None, 1=Student, 2=Working}

External Factor
30: Weather {0=Fine, 1=Cloudy, 2=Rainy}
31: Temperature (Celsius) {-5 to 40}


User profiles:

user_id age gender
user01   26  male
user02   24  male
user03   24  male
user04   34  male
user05   25  male
user06   24  male
user07   25  male
user08   25  male
