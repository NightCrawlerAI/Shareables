
import os
import openai
import requests
from bs4 import BeautifulSoup
from scraper_api import ScraperAPIClient
import time
from thefuzz import fuzz
import logging
import pandas
import sys

openai.api_key = "######"

# Initialize the ScraperAPI client with your API key
client = ScraperAPIClient('#######')


def is_duplicate(content, processed_articles):
    similarity_threshold = 60
    for index, processed_article in enumerate(processed_articles):
        similarity_score = fuzz.token_set_ratio(content, processed_article)
        print(f"Similarity Score: {similarity_score}")
        if similarity_score > similarity_threshold:
            return similarity_score, index
    return None, None


def generate_results(article_links, prompt_links):
    analysis_results = []
    processed_articles = []
    article_titles = []
    similar_article_pairs = []
    new_article_info = []

    for article_link in article_links:
        article_title, article_text = scraper(article_link)
        if article_text is None:
            continue

        similarity_score, similar_index = is_duplicate(article_text, processed_articles)
        if similarity_score is not None:
            similar_article_pairs.append(
                (article_titles[similar_index], article_links[similar_index], article_title, article_link))
            print(f"Similar article found: {article_link}. Skipping...\n")
        else:
            processed_articles.append(article_text)
            article_titles.append(article_title)
            print(f"Processed {article_title}\n")

            # Perform analysis for unique articles
            analysis, token_error = ai_analyzer(article_text, prompt_links)
            article_info = f"{article_title} | <a href='{article_link}'>Link</a>"
            print(article_info)
            if not token_error:
                new_article_info.append(article_info)
            analysis_results.extend(analysis)

    # List similar articles
    if similar_article_pairs:
        print("Similar Articles:")
        for article_1_title, article_1_link, article_2_title, article_2_link in similar_article_pairs:
            print(f"- {article_1_title} ({article_1_link})")
            print(f"  is similar to:")
            print(f"- {article_2_title} ({article_2_link})\n")

    time.sleep(5)
    print(similar_article_pairs, "\n")
    print(processed_articles)
    return analysis_results, new_article_info


# Collect raw text from webpage/article with the bs4 scraper
# This is to be fed into the openai function
def scraper(article):
    try:
        try:
            other_response = requests.get(article)
            article_url = other_response.url
            response = client.get(url=article_url)
            print("scraperapi", response.url)
            # print(response.text)
        except Exception as e:
            logging.error(f"Error occurred during request: {str(e)}")
            return None

        info = BeautifulSoup(response.text, "html.parser")
        # print(info)

        # Check if <h1> tag is found
        h1_tag = info.find('h1')
        # print(h1_tag)
        if h1_tag:
            article_title = h1_tag.text.strip()
        else:
            print(f"No title found for {article}. Using default title...\n")
            article_title = "Default title"

        article_content = ""
        paragraphs = info.find_all("p")
        for paragraph in paragraphs:
            article_content += paragraph.text.strip() + "\n"

        if "404" in article_title or "403" in article_title or "blocked" in article_title:
            logging.error(f"The page {article} returned a {article_title} error. Skipping...")
            return None
        logging.info(f"Processing {article_title}")

        return article_title, article_content

    except Exception as e:
        print(f"An error occurred while processing {article}: {e}")
        return None, None


# def is_useful(content):
#     # Criteria 1: Length of the content
#     if len(content) < 75:  # 100 is an arbitrary number; you can set this threshold as per your requirement
#         return False
#
#     # Criteria 2: Presence of specific keywords that indicate useful information
#     useful_keywords = ['trade', 'investment', 'prices', 'demand', 'exports', 'agreement', 'volume']
#     useful_keywords_present = any(keyword in content.lower() for keyword in useful_keywords)
#
#     # Criteria 3: Absence of keywords/phrases that indicate lack of information
#     non_useful_phrases = ['difficult to predict', 'does not provide any information']
#     non_useful_phrases_present = any(phrase in content.lower() for phrase in non_useful_phrases)
#
#     # Evaluate the usefulness based on the criteria weights
#     if useful_keywords_present and not non_useful_phrases_present:
#         return True
#     elif useful_keywords_present and non_useful_phrases_present:
#         return True
#     else:
#         return False


# This function is using chat completion
# to answer pre-determined questions about the article
def ai_analyzer(content, prompts):
    ai_conclusions = []
    analysis = ""
    token_error = False
    # print(content)
    # print(prompts)
    for prompt in prompts:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": f"Your role is to memorize this article{content}."},
                    {"role": "user", "content": prompt}
                    # {"role": "system", "content": f"{analysis}"}
                ],
                max_tokens=300,  # Set the desired maximum length of the response
                temperature=0.4,  # Adjust the temperature to control the randomness of the output
                n=3,  # Set the number of completions to generate
                stop=None,  # Specify a stop sequence to control the length of the response
            )
            analysis = response['choices'][0]['message']['content']
            ai_conclusions.append(analysis)

        except openai.error.InvalidRequestError as e:
            # Skip the current prompt if content is too long
            token_error = True

    return ai_conclusions, token_error

