Music Genre Prediction - First Steps
 
 Poster: https://tartuulikool-my.sharepoint.com/personal/victor88_ut_ee/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fvictor88%5Fut%5Fee%2FDocuments%2Fvideos%20ids%20projects%202021%2Fd18%2Epdf&parent=%2Fpersonal%2Fvictor88%5Fut%5Fee%2FDocuments%2Fvideos%20ids%20projects%202021&ga=1

Introduction to Data Science

Project title: **Music Genre Prediction**


Team members:

- *Savelii Vorontcov*
- *Eduard Žurin*
- *Tatjana Ratilovskaja*

This report consists of three parts: Business Understanding, Data Understanding and Project Planning. According to CRISP-DM, Project Planning is a part of Business Understanding, but in this report it’s presented separately.

1. Business Understanding

Business Goals

Background

Our project is a part of the course Introduction to Data Science (LTAT.02.002). Our team can be considered as a side of interest and our goal can be defined as learning a lot about Data Science.

However, since our dataset was originally derived using Spotify API, Spotify could be considered a client company in this situation. Spotify is a for-profit publicly traded company, so the primary interest of the company and its stakeholders may be increasing total revenue or net income through improving the quality of the product.

Business Goals

Since it is a student project, we don’t have a specific business goal. We can define one as getting a lot of first-hand knowledge in Data Science.

In case Spotify was our client, a business goal could have been to increase total annual revenue (or net income) by, for example, 0.1%. Current revenue is $9 billion, so that would make a $9 million increase.

Business Success Criteria

For a student team, criteria can be defined quantitatively as increasing total points of each project member in the course by 20 (maximum points for the project) or qualitatively by making a good model (in that case, “goodness” of our model can be evaluated by course TA’s).

In a real business case scenario, results can be measured by calculating total revenue (or net income) during some period of time after our work has taken place. Other values that are influenced by our work directly can be calculated (for example, customers can evaluate how good genres are predicted using our system compared to previous systems, or, in case we improve app’s recommendation system, rate of how many songs customers skip before and after our work has taken place). Then a correlation between business values (such as revenue) and direct metrics (such as amount of song skips or amount of likes) can be calculated.

Assessment of Situation

Inventory of resources

- Data: music genre dataset composed from audio features available on Spotify.
- Software resources: Jupyter Notebook, GitHub, git, Python 3, Anaconda, Keras/Tensorflow, Tableau.
- People: three students specializing in Computer Science and familiar with Data Science.
- Hardware: at least three personal computers, including desktops and laptops.
- Possibly Spotify API (more data can be obtained).

Requirements, assumptions, and constraints

Requirements:

- Trained and tested genre prediction model(s) with high accuracy.
- Presentational poster that describes the work that we have done and the results achieved.
- GitHub repository including all the code and readme-files describing the files and how to run the code.
- A 3-minute video with project introduction.

Assumptions:

- Data in the dataset is available to use for research purposes.
- Each team member updates the repository after they have made changes in the files.

Constraints:

- All the work on the project, including 3-minute video and poster, must be concluded and submitted by December 13-th.
- Project presentation is held on December 16-th.

Risks and contingencies

- If some of our teammates are unable to use their hardware or software, we can organize video call sessions or in-person meetings where we work together, so that everyone is able to access the files and contribute to the project.
- If data in the dataset turns out to be insufficient, we are ready to collect additional features or gather more instances for the existing one using Spotify API.
- There is a very small chance that access to GitHub might be interrupted at some points of time, but if that happens, every project teammate has a local repo of the project, since we are using a version control system.

Terminology

Accuracy - proportion of correct predictions made by the prediction model.

Error rate - proportion of wrong predictions made by the prediction model.

Precision - proportion of true positives among predicted positives.

Recall - proportion of true positives among actual positives.

Label - attribute that we want to predict (music genre in our case).

Training instance - any row for which the value of the label is known, used to train the

prediction model. All training instances combined make up the training data.

Test instance - any row for which the value of the label is unknown, used to test predictions made by the model. All test instances combined make up the test data.

Costs and benefits

Costs: No monetary expenses are required from the team members in order to successfully finish the project.

Each team member is expected to spend at least 30 hours of work on the project, amounting to 90 hours in total from all 3 members.

Benefits: Project completion and presentation amount to a maximum of 10 points in the course Introduction to Data Science (LTAT.02.002) and is compulsory in order to pass the course.

Data Mining / Data Analysis / Machine Learning Goals

Data-mining goals

Our goal is to train different types of models on the dataset which predict music genre from features. Based on the created models we want to achieve the best prediction measures. We also aim to find interesting correlations (genre + duration, genre + popularity, key + popularity, etc).

We plan to document all the findings and deliver reports with them. On top of that, we plan to deliver visualizations of all interesting findings.

Data-mining success criteria

Our main goal is to train accurate genre prediction model. We are successful if our prediction model accuracy score will be greater than 0.8. Apart from accuracy, we plan to verify the model using other measures such as f1-score and possibly ROC-AUC scores.

2. Data Understanding

Gathering Data

Data Requirements

The only data type we need is tabular data on songs. Such data includes metadata in some sense, since it doesn’t contain any information about song contents (music itself). Time of data gathering generally does not matter, but we assume it should not be more than 2 years old. Dataset that we are planning to use was published in October this year and entries were gathered in April, so we assume it was gathered this year, and therefore, fits our requirements.

Data Availability

Data to be analyzed is gathered in a form of existing dataset published on Kaggle platform. The dataset is in open access and is distributed under CC-0 licence (public domain) and everyone registered on Kaggle can download it. In case we need more data, we can use Spotify API (original dataset was gathered using this API). We haven’t tested the Spotify API yet, since we assume that the gathered dataset will provide us enough insights.

Selection Criteria

In terms of memory, the dataset can be used as a whole, because its size is small enough to fit into RAM (~8MB). Each entry in the dataset has 18 fields - we are planning to use about 15 of them (non-unique data for each song). We do not plan to use instance ids, track names and artist names, since those attributes are too subjective for each song and can cause overfitting. We do not plan to use dates when an instance was obtained, since it won’t be useful for the model. In case of missing or outlier attributes, we plan to use imputing techniques. We also plan to perform data normalization and scaling.

Describing Data

Dataset is stored in a .csv file. There are 41700 tracks in it, each with a set of features taken from the Spotify database:

- **instance\_id** - unique identifier for each music track.
- **artist\_name** - artist name.
- **track\_name** - track name.
- **popularity** - track popularity on a scale from 0 to 99.
- **acousticness** - confidence measure from 0.0 to 1.0 of whether the track is acoustic.
- **danceability** - how suitable a track is for dancing on a scale from 0.0 to 1.0, based on a combination of musical elements.
- **duration\_ms** - duration of the track in milliseconds.
- **energy** - a perceptual measure of intensity and activity on a scale from 0.0 to 1.0. For example, death metal is considered to have high energy, while ballads low.
- **instrumentalness** - prediction whether a track contains no vocals on a scale from

0.0 to 1.0.

- **key** - music key that the track is written in: C, C#, D, …, B.
- **liveness** - presence of an audience in the recording on a scale from 0.00 to 1.0, where values above 0.8 means high confidence that the track is a live recording.
- **loudness** - loudness of a track in decibels averaged across the entire track. Values typically range from -60 to 0.
- **mode** - whether a track is major (1) or minor (0).
- **speechiness** - presence of spoken words in a track on a scale from 0.0 to 1.0. Values above 0.66 show that track is mostly speech and not music, values between
  - and 0.66 show that track contains a lot of both (e.g., rap tracks).
- **tempo** - tempo of a track in beats per minute (BPM).
- **obtained\_date** - date when the track was obtained.
- **valence** - musical positiveness conveyed by a track on a scale from 0.0 to 1.0, where higher values show that the track is more happy sounding and vice versa.
- **music\_gence** - genre assigned to the track by Spotify.

Exploring Data

When having a closer look at our dataset I can already see that some columns have undefined values, therefore some manipulation is needed before any further handling. I can also see that "obtained\_date" column shows data obtaining date. This makes this column pretty much useless, because we don't need that information. Columns: "acousticness", "danceability", "energy", "instrumentalness", "liveness", "speechiness", "valence" have values in the range from 0 to 1 so we need to change that range from -1 to 1. Column “loudness” has values in decibels, so it may require some transformations.

Verifying Data Quality

Generally, data looks good enough to do analysis and train prediction models. Some entries have missing attributes, but those can be overcome by either imputing techniques. Some entries may have outlier attributes - such entries can also be imputed or excluded from analysis altogether. We prefer the former, since every entry may be valuable in terms of other attributes.

Some data may exist, but not be present in our dataset. Such data can be obtained using Spotify API.

3. Project Planning

Project Plan

Initial plan:

1. First two steps according to CRISP-DM: business understanding and data understanding (deadline - 29.11).
1. Initial data preprocessing, initial model training, finding dependencies and visualizing (deadline - 03.12).
1. Full data preprocessing (incl. normalization/scaling), more model training (incl. cross-validation and deep learning models), more visualizing of data (deadline - 10.12).
1. Preparing final report, presentation and video with project overview (deadline - 13.12).
1. Present project on poster session (16.12).

Tasks:

- First steps of the project (HW10) (3 hours for each team member, 9 in total)
- Data cleaning (imputing, removing entries) (Eduard, 9 hours)
- Data normalization (scaling, one-hot encoding) (Eduard, 7 hours)
- Dimensionality reduction (PCA, t-SNE) (Eduard, 8 hours)
- Data analysis on the dataset (Tatjana, 10 hours)
- Data mining on the dataset (Tatjana, 6 hours)
- Creating genre prediction models with and without Deep Learning (Savelii, 10 hours)
- Measuring model quality and visualizing it (Savelii, 6 hours)
- Documenting and reporting all key findings (2 hours for each team member)
- Visualize all findings so they would be understandable (Tatjana, 8 hours)
- Making of the video (3 hours for each team member, 9 in total)
- Possibly trying simple techniques for music recommendation (such as nearest neighbors or vector embeddings) (Savelii, 4 hours)

Methods and Tools

Tools should include tools for Data Science, which are, in fact, freely available to us. Such tools include programming language with numerical computation support (Python + numpy), tabular data support (pandas), frameworks for machine learning (scikit-learn, Keras/Tensorflow), visualization support (matplotlib, seaborn), tools for model persistence (joblib), tools for statistical analysis (scipy).

In case of doing something more advanced, we can use more specialized tools (such as tableau for visualizing).
