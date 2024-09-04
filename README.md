# Twitter Crawling Program

This is a simple program designed to crawl tweets from Twitter based on specific keywords. The program collects tweets and stores them for further analysis or processing.

## Features

- Crawl tweets based on one or more keywords.
- Extract relevant tweet information, such as date, username, and tweet content.
- Save the extracted tweets to a CSV or Excel file.
- Simple and easy-to-use interface.

## Requirements

- Python 3.x
- Tweepy (Twitter API library)
- Pandas (for data manipulation)
- Openpyxl (for Excel file handling)

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/username/twitter-crawling.git
    cd twitter-crawling
    ```

2. **Install Required Libraries**:
    Make sure you have Python 3 installed. Then install the required Python packages using pip:
    ```bash
    pip install tweepy pandas openpyxl
    ```

3. **Setup Twitter API Credentials**:
   - Sign up for a [Twitter Developer Account](https://developer.twitter.com/).
   - Create a new app in the Twitter Developer Portal and get the following credentials:
     - `API Key`
     - `API Secret Key`
     - `Access Token`
     - `Access Token Secret`
   - Add your Twitter API credentials to a `.env` file or directly in the script.

## Usage

1. **Configure the Script**:
    Open the script and set your Twitter API credentials and the keyword(s) you want to search for.
    ```python
    CONSUMER_KEY = 'your-consumer-key'
    CONSUMER_SECRET = 'your-consumer-secret'
    ACCESS_TOKEN = 'your-access-token'
    ACCESS_TOKEN_SECRET = 'your-access-token-secret'
    
    keyword = 'your-search-keyword'
    ```

2. **Run the Program**:
    ```bash
    python twitter_crawl.py
    ```

3. **Output**:
    The program will generate a CSV or Excel file (`tweets.csv` or `tweets.xlsx`) containing the following columns:
    - `DATE` - The date when the tweet was posted.
    - `USER NAME` - The username of the person who posted the tweet.
    - `TWEET CONTENT` - The content of the tweet.

## Example

To search for tweets containing the keyword "Python", configure the script as follows:

```python
keyword = 'Python'
```

Run the script:

```bash
python twitter_crawl.py
```

The output will be a file named `tweets.csv` or `tweets.xlsx`, with all tweets containing the keyword "Python."

