### Version 2.1 (Final)
import re
import sys

import requests
from pygooglenews import GoogleNews
from datetime import datetime as dtdt
from datetime import timedelta
import win32com.client as win32
import datetime
import pandas as pd
import warnings
from nltk import word_tokenize
from nltk.corpus import stopwords
from thefuzz import fuzz
import time

import news_analysis
from news_analysis import generate_results
import nltk
import difflib

to_email = "trading@e360power.com"
bcc_email = "kjones@e360power.com"#; schakraborty@e360power.com

warnings.simplefilter(action='ignore', category=FutureWarning)
try:
    # Get today's date and time
    today = dtdt.combine(dtdt.today(), dtdt.min.time())

    # Use the GoogleNews API to search for articles on natural gas trading
    gn = GoogleNews()
    # test = gn.topic_headlines('CAAqBwgKMOG-iAswlKaHAw')
    if dtdt.now() > dtdt.now().replace(hour=12, minute=0, second=0, microsecond=0):
        test = gn.search('allintext:natural+gas OR natural+gas+trading OR henry+hub OR pipeline', when='5h')
    else:
        test = gn.search('allintext:natural+gas OR natural+gas+trading OR henry+hub OR pipeline', when='19h')

    # Extract the articles
    articles = test['entries']
    print('Nat Gas News Scraper v2.3', len(articles))
    # raise KeyboardInterrupt
    # Convert the articles to a pandas dataframe
    base = pd.json_normalize(articles)
    base['published'] = pd.to_datetime(base.published, format='%a, %d %b %Y %X GMT')
    base['published'] = base.published.dt.to_pydatetime()

    # Only keep articles published today or later
    final_base = base[base.published >= today].copy()

    # Sort the articles by publication date, most recent first
    final_base.sort_values('published', inplace=True, ascending=False)
    final_base.reset_index(drop=True, inplace=True)

    # Define a list of keywords that are relevant to natural gas trading
    keywords = ['natural gas', 'trading', 'trade', 'price', 'market', 'supply', 'demand', 'energy', 'lng', 'gas',
                'dominion', 'algonquin',
                'ferc', 'freeport', 'pipeline', 'outage', 'eia', 'texas', 'henry hub', 'henry', 'oil', 'crude',
                'capacity', 'congestion',
                'reserves', 'consumption', 'storage', 'hdd', 'ldd', 'export', 'import', 'renewables', 'coal',
                'drilling', 'hgl', 'inventory',
                'inventories', 'production', 'ethane', 'wti', 'exploration', 'forecast', 'congress', 'congressional',
                'electricity', 'power',
                'withdrawal', 'withdrawals', 'injection', 'injections', 'futures', 'pacific', 'mountain', 'citygate',
                'basis', 'hub',
                'south central', 'salt', 'non-salt', 'non salt', 'east', 'midwest', 'weather', 'balances', 'balance',
                'volatility',
                'interest rates', 'interest', 'rate', 'wind', 'degrees', 'rescomm', 'permian', 'osha', 'disruption',
                'pipe', 'explosion', 'fault', 'nymex', 'cme', 'ttf', 'boil', 'proshares', 'bloomberg', 'kold', 'ung',
                'hnu', 'hnd', 'boil', 'leak']

    # Remove stop words and tokenize the article titles
    stop_words = set(stopwords.words('english'))
    titles = final_base['title'].str.lower().apply(
        lambda x: [word for word in word_tokenize(x) if word not in stop_words])

    # Compute a relevance score for each article based on the number of keywords it contains
    relevance_scores = []
    for title in titles:
        score = len(set(title).intersection(keywords))
        relevance_scores.append(score)

    # Add the relevance scores to the dataframe
    final_base['relevance_score'] = relevance_scores
    print(final_base)

    # Sort the articles by relevance score, highest first
    final_base.sort_values('relevance_score', inplace=True, ascending=False)
    final_base.reset_index(drop=True, inplace=True)
    final_base = final_base[final_base.relevance_score >= 2].copy()
    print(final_base)
    # Extract the relevant articles
    # Create 'counter' and 'article_info' columns in final_base
    final_base['counter'] = range(1, len(final_base) + 1)
    final_base['article_info'] = final_base.apply(
        lambda x: f"{x['counter']}. {x['title']} | {x['published']} | <a href='{x['link']}'>Link</a>", axis=1)
    print(final_base)
    # Get links and prompts
    links = final_base['source.href'].tolist()
    links_value_pair = dict(zip(final_base['source.href'], final_base['link']))
    # raise Exception
    new_links = []
    # counter = 0
    for key, value in links_value_pair.items():
        # time.sleep(1)
        # new_link = requests.get(link, headers=
        # {"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"}).url
        print(key)
        # Ignore sources that require reCAPTCHA or Login
        ignored_sources = ['bloomberg', 'marketscreener', 'nnn', 'rbnenergy', 'nasdaq', 'qcintel', 'xm']
        if any(ignored_source in key for ignored_source in ignored_sources):
            continue
        else:
           new_links.append(value)
        # new_link = final_base['link']

        # counter += 1
    print(new_links)

    prompts = ["Summarize this in max 30 words.",
               "What are potential market impacts for now and the future. Keep it max 30 words.",
               "Give a bullish or bearish indication for natural gas. Answer in max one word."]

    # Generate analysis
    analysis, new_article_info = news_analysis.generate_results(new_links, prompts)
    analysis = [text.replace('\n', '<br>') for text in analysis]
    print(len(analysis))
    # Reshape analysis
    reshaped_analysis = [
        (analysis[i], analysis[i + 1], analysis[i + 2])
        for i in range(0, len(analysis), 3)
    ]

    print(reshaped_analysis)


    # Create a function to apply the styling to the DataFrame's HTML
    def style_cells(html):
        html = re.sub(r'>(B|b)ullish\.?<', r' style="background-color: lightgreen">\1ullish<', html)
        html = re.sub(r'>(B|b)earish\.?<', r' style="background-color: lightcoral">\1earish<', html)
        return html


    # Create DataFrame
    columns = prompts
    analysis_dataframe = pd.DataFrame(reshaped_analysis, columns=columns)
    analysis_dataframe.insert(0, "Article Info", new_article_info)
    column_rename_mapping = {
        "Summarize this in max 30 words.": "Summary",
        "What are potential market impacts for now and the future. Keep it max 30 words.": "Market Impact",
        "Give a bullish or bearish indication for natural gas. Answer in max one word.": "Indication"
    }

    # Rename the columns
    analysis_dataframe.rename(columns=column_rename_mapping, inplace=True)
    # Convert DataFrame to HTML
    analysis_html = analysis_dataframe.to_html(index=False, escape=False)

    # Apply styling to HTML
    analysis_html = style_cells(analysis_html)

    # Display the DataFrame
    # print(analysis_dataframe)
    email_text = '<br>'.join(final_base['article_info'].tolist())
    email_html = f"""
    <h3>Nat Gas News Scraper v2.3</h3>
    <p>{email_text}</p>
    <h3>Analysis</h3>
    {analysis_html}
    """


    # send email
    def Email(html_content):
        outlook = win32.Dispatch('outlook.application')
        mail = outlook.CreateItem(0)
        mail.To = to_email
        mail.BCC = bcc_email
        mail.Subject = "Nat Gas News Scraper v2.3"
        mail.HTMLBody = html_content  # Setting HTMLBody instead of Body
        mail.Send()


    print(final_base)
    if len(final_base) == 0:
        print("email not sent")
        pass
    else:
        Email(email_html)
        print('Email Sent!')
except AttributeError:
    pass


# import sys
# sys.exit(1)

try:
    ### ETF News Scraper
    today = dtdt.combine(dtdt.today(), dtdt.min.time())

    gn = GoogleNews()

    if dtdt.now() > dtdt.now().replace(hour=12, minute=0, second=0, microsecond=0):
        test = gn.search('allintext:boil+etf OR boil+natural gas OR kold+etf OR kold+natural gas OR ung+etf OR ung+natural gas \
                        OR hnu+natural gas OR hnu+etf OR hnd+natural gas OR hnd+etf OR natural gas+etf', when='5h')
    else:
        test = gn.search('allintext:boil+etf OR boil+natural gas OR kold+etf OR kold+natural gas OR ung+etf OR ung+natural gas \
                        OR hnu+natural gas OR hnu+etf OR hnd+natural gas OR hnd+etf OR natural gas+etf', when='19h')

    # Extract the articles
    articles = test['entries']
    print('Nat Gas ETF News Scraper', len(articles))

    # Convert the articles to a pandas dataframe
    base = pd.json_normalize(articles)
    base['published'] = pd.to_datetime(base.published, format='%a, %d %b %Y %X GMT')
    base['published'] = base.published.dt.to_pydatetime()

    # Only keep articles published today or later
    final_base = base[base.published >= today].copy()

    # Sort the articles by publication date, most recent first
    final_base.sort_values('published', inplace=True, ascending=False)
    final_base.reset_index(drop=True, inplace=True)

    # Define a list of keywords that are relevant to natural gas trading
    keywords = ['natural gas', 'trading', 'trade', 'price', 'market', 'supply', 'demand', 'energy', 'lng', 'gas',
                'dominion', 'algonquin',
                'ferc', 'freeport', 'pipeline', 'outage', 'eia', 'texas', 'henry hub', 'henry', 'oil', 'crude',
                'capacity', 'congestion',
                'reserves', 'consumption', 'storage', 'hdd', 'ldd', 'export', 'import', 'renewables', 'coal',
                'drilling', 'hgl', 'inventory',
                'inventories', 'production', 'ethane', 'wti', 'exploration', 'forecast', 'congress', 'congressional',
                'electricity', 'power',
                'withdrawal', 'withdrawals', 'injection', 'injections', 'futures', 'pacific', 'mountain', 'citygate',
                'basis', 'hub',
                'south central', 'salt', 'non-salt', 'non salt', 'east', 'midwest', 'weather', 'balances', 'balance',
                'volatility',
                'interest rates', 'interest', 'rate', 'wind', 'degrees', 'rescomm', 'permian', 'osha', 'disruption',
                'pipe', 'explosion', 'fault', 'nymex', 'cme', 'ttf', 'boil', 'proshares', 'bloomberg', 'kold', 'ung',
                'hnu', 'hnd', 'boil',
                'ultrashort', 'betapro', 'etf']

    # Remove stop words and tokenize the article titles
    stop_words = set(stopwords.words('english'))
    titles = final_base['title'].str.lower().apply(
        lambda x: [word for word in word_tokenize(x) if word not in stop_words])

    # Compute a relevance score for each article based on the number of keywords it contains
    relevance_scores = []
    for title in titles:
        score = len(set(title).intersection(keywords))
        relevance_scores.append(score)

    # Add the relevance scores to the dataframe
    final_base['relevance_score'] = relevance_scores

    # Sort the articles by relevance score, highest first
    final_base.sort_values('relevance_score', inplace=True, ascending=False)
    final_base.reset_index(drop=True, inplace=True)
    final_base = final_base[final_base.relevance_score >= 2].copy()

    # Extract the relevant articles
    final_base['counter'] = range(1, len(final_base) + 1)
    final_base['article_info'] = final_base.apply(
        lambda x: f"{x['counter']}. {x['title']} | {x['published']} | {x['link']}", axis=1)
    email_text = '\n'.join(final_base['article_info'].tolist())


    # send email
    def Email(text):
        outlook = win32.Dispatch('outlook.application')
        mail = outlook.CreateItem(0)
        mail.To = to_email
        mail.BCC = bcc_email
        mail.Subject = "Nat Gas ETF News Scraper v2.3"
        mail.Body = text
        mail.Send()


    print(final_base)
    if len(final_base) == 0:
        pass
    else:
        Email(email_text)
        print('Email Sent!')
except AttributeError:
    pass

try:
    ###Washington Carbon Scraper
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Get today's date and time
    today = dtdt.combine(dtdt.today(), dtdt.min.time())

    # Use the GoogleNews API to search for articles on natural gas trading
    gn = GoogleNews()
    # test = gn.topic_headlines('CAAqBwgKMOG-iAswlKaHAw')
    if dtdt.now() > dtdt.now().replace(hour=12, minute=0, second=0, microsecond=0):
        test = gn.search(
            'allintext:washington+carbon OR washington+auction OR washington+emissions+law OR california+carbon OR carbon+allowance OR washington+ecology OR california+air+resources+board',
            when='5h')
    else:
        test = gn.search(
            'allintext:washington+carbon OR washington+auction OR washington+emissions+law OR california+carbon OR carbon+allowance OR washington+ecology OR california+air+resources+board',
            when='19h')

    # Extract the articles
    articles = test['entries']
    print('Washington Carbon Scraper ', len(articles))

    # Convert the articles to a pandas dataframe
    base = pd.json_normalize(articles)
    base['published'] = pd.to_datetime(base.published, format='%a, %d %b %Y %X GMT')
    base['published'] = base.published.dt.to_pydatetime()

    # Only keep articles published today or later
    final_base = base[base.published >= today].copy()

    # Sort the articles by publication date, most recent first
    final_base.sort_values('published', inplace=True, ascending=False)
    final_base.reset_index(drop=True, inplace=True)

    keywords = ['washington', 'carbon', 'emissions', 'cap', 'pollution', 'credit', 'allowance', 'law',
                'auction', 'price', 'washington carbon', 'cca', 'california', 'california carbon',
                'california carbon allowance', 'california carbon credit', 'california carbon cap',
                'california air resources board', 'carb', 'arb', 'wci', 'western climate initiative',
                'regional greenhouse gas initiative', 'rggi', 'climate', 'climate change', 'greenhouse gas',
                'pennsylvania', 'pennsylvania carbon', 'pennsylvania carbon allowance', 'pennsylvania carbon credit',
                'west virginia', 'west virginia carbon', 'west virginia carbon allowance',
                'west virginia carbon credit']

    # Remove stop words and tokenize the article titles
    stop_words = set(stopwords.words('english'))
    titles = final_base['title'].str.lower().apply(
        lambda x: [word for word in word_tokenize(x) if word not in stop_words])

    # Compute a relevance score for each article based on the number of keywords it contains
    relevance_scores = []
    for title in titles:
        score = len(set(title).intersection(keywords))
        relevance_scores.append(score)

    # Add the relevance scores to the dataframe
    final_base['relevance_score'] = relevance_scores

    # Sort the articles by relevance score, highest first
    final_base.sort_values('relevance_score', inplace=True, ascending=False)
    final_base.reset_index(drop=True, inplace=True)
    final_base = final_base[final_base.relevance_score >= 2].copy()

    # Extract the relevant articles
    final_base['counter'] = range(1, len(final_base) + 1)
    final_base['article_info'] = final_base.apply(
        lambda x: f"{x['counter']}. {x['title']} | {x['published']} | <a href='{x['link']}'>Link</a>", axis=1)
    print(final_base)
    # Get links and prompts
    links = final_base['source.href'].tolist()
    links_value_pair = dict(zip(final_base['source.href'], final_base['link']))
    # raise Exception
    new_links = []
    # counter = 0
    for key, value in links_value_pair.items():
        print(key)
        ignored_sources = ['bloomberg', 'marketscreener', 'nnn']
        if any(ignored_source in key for ignored_source in ignored_sources):
            continue
        else:
            new_links.append(value)
        # new_link = final_base['link']

        # counter += 1
    print(new_links)

    prompts = ["Summarize this in max 30 words.",
               "What are potential market impacts for now and the future. Keep it max 30 words.",
               "Give a bullish or bearish indication for natural gas. Answer in max one word."]

    # Generate analysis
    analysis, new_article_info = news_analysis.generate_results(new_links, prompts)
    analysis = [text.replace('\n', '<br>') for text in analysis]
    print(len(analysis))
    # Reshape analysis
    reshaped_analysis = [
        (analysis[i], analysis[i + 1], analysis[i + 2])
        for i in range(0, len(analysis), 3)
    ]

    print(reshaped_analysis)


    # Create a function to apply the styling to the DataFrame's HTML
    def style_cells(html):
        html = re.sub(r'>(B|b)ullish\.?<', r' style="background-color: lightgreen">\1ullish<', html)
        html = re.sub(r'>(B|b)earish\.?<', r' style="background-color: lightcoral">\1earish<', html)
        return html


    # Create DataFrame
    columns = prompts
    analysis_dataframe = pd.DataFrame(reshaped_analysis, columns=columns)
    analysis_dataframe.insert(0, "Article Info", new_article_info)
    column_rename_mapping = {
        "Summarize this in max 30 words.": "Summary",
        "What are potential market impacts for now and the future. Keep it max 30 words.": "Market Impact",
        "Give a bullish or bearish indication for natural gas. Answer in max one word.": "Indication"
    }

    # Rename the columns
    analysis_dataframe.rename(columns=column_rename_mapping, inplace=True)
    # Convert DataFrame to HTML
    analysis_html = analysis_dataframe.to_html(index=False, escape=False)

    # Apply styling to HTML
    analysis_html = style_cells(analysis_html)

    # Display the DataFrame
    # print(analysis_dataframe)
    email_text = '<br>'.join(final_base['article_info'].tolist())
    email_html = f"""
        <h3>Washington Carbon Scraper v2.3</h3>
        <p>{email_text}</p>
        <h3>Analysis</h3>
        {analysis_html}
        """


    # send email
    def Email(html_content):
        outlook = win32.Dispatch('outlook.application')
        mail = outlook.CreateItem(0)
        mail.To = to_email
        mail.BCC = bcc_email
        mail.Subject = "Washington Carbon Scraper v2.3"
        mail.HTMLBody = html_content  # Setting HTMLBody instead of Body
        mail.Send()


    print(final_base)
    if len(final_base) == 0:

        print("email not sent")
        pass
    else:
        Email(email_html)
        print('Email Sent!')
except AttributeError:
    pass

#
try:
    ###Mountain Valley Pipeline Scraper
    from serpapi import GoogleSearch
    from datetime import datetime as dtdt
    from datetime import timedelta as td
    import pandas as pd
    import win32com.client as win32
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    # from flask import Flask, render_template, request
    from fuzzymatcher import fuzzy_left_join
    import pythoncom
    import dateparser


    def fetch_news(search_query, keywords):
        params = {
            "engine": "google",
            "q": search_query,
            "gl": 'us',
            'hl': 'en',
            'tbm': 'nws',
            "api_key": "f4345351a965f1adea7ce426f2b7445ecb9e06126307cca348d5e0332a986d53"
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        #     display(results)

        # Check if 'news_results' is in the dictionary
        news_results = results.get('news_results', None)
        if news_results is None:
            print("No news_results found in the results")
            return

        df = pd.json_normalize(news_results)

        #     display(df)

        def convert_date(x):
            if 'ago' in x:
                if 'min' in x:
                    return dtdt.now() - td(minutes=int(x.split()[0]))
                elif 'hour' in x:
                    return dtdt.now() - td(hours=int(x.split()[0]))
                elif 'day' in x:
                    return dtdt.now() - td(days=int(x.split()[0]))
                else:
                    return pd.NaT
            else:
                try:
                    return dtdt.strptime(x, "%b %d, %Y")
                except ValueError:
                    return pd.NaT

        #     for i in range(len(df.date)):
        #         print('\n',convert_date(df['date'][i]),'\n')
        #     print(convert_date(df['date'][8]))

        df['published'] = df['date'].apply(convert_date)
        print('after converting dates')

        df['published_date'] = df['published']  # .dt.date
        print('created new publish date column')

        today = dtdt.now()  # today.date()
        if dtdt.now() > dtdt.now().replace(hour=12, minute=0, second=0, microsecond=0):
            yesterday = today - td(hours=5)
        else:
            yesterday = today - td(days=1)

        df_today = df[(df.published_date >= yesterday) & (df.published_date <= today)].copy()
        print('after filtering for published date today')

        stop_words = set(stopwords.words('english'))

        df_today['title_words'] = df_today['snippet'].str.lower().apply(
            lambda x: [word for word in word_tokenize(x) if word not in stop_words])

        df_today['relevance_score'] = df_today.title_words.apply(lambda x: len(set(x).intersection(keywords)))
        df_today.sort_values('relevance_score', inplace=True, ascending=False)
        print('After creating Relevance Score column')

        df_today = df_today[df_today.relevance_score >= 2].copy()
        print('After Filtering for Relevance Scoring')

        df_today['counter'] = range(1, len(df_today) + 1)
        df_today['article_info'] = df_today.apply(
            lambda x: f"{x['counter']}. {x['title']} | {x['published']} | <a href='{x['link']}'>Link</a>", axis=1)

        print('Final Output')

        # Get links and prompts
        links = df_today['link'].tolist()
        links_value_pair = dict(zip(df_today['source'], df_today['link']))
        # raise Exception
        new_links = []
        # counter = 0
        for key, value in links_value_pair.items():
            # time.sleep(1)
            # new_link = requests.get(link, headers=
            # {"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"}).url
            print(key)
            ignored_sources = ['bloomberg', 'marketscreener', 'nnn', 'bizjournals']#, 'reuters', 'wsj', 'yahoo', 'marketwatch',]
            if any(ignored_source in value for ignored_source in ignored_sources):
                continue
            else:
                new_links.append(value)
            # new_link = final_base['link']

            # counter += 1
        print(new_links)
        prompts = ["Summarize briefly in 20 to 30 words.",
                   "Itemize potential market impacts in 2023, 2024, 2025. Keep it short",
                   "Itemize potential impacts from 2025 onwards if any. Keep it short"]

        # Generate analysis
        analysis, new_article_info = news_analysis.generate_results(new_links, prompts)
        analysis = [text.replace('\n', '<br>') for text in analysis]
        print(len(analysis))
        # Reshape analysis
        reshaped_analysis = [
            (analysis[i], analysis[i + 1], analysis[i + 2])
            for i in range(0, len(analysis), 3)
        ]

        print(reshaped_analysis)

        # Create DataFrame
        columns = prompts
        print(columns)
        analysis_dataframe = pd.DataFrame(reshaped_analysis, columns=columns)
        analysis_dataframe.insert(0, "Article Info", new_article_info)
        column_rename_mapping = {
            "Summarize briefly in 20 to 30 words.": "Summary",
            "Itemize potential market impacts in 2023, 2024, 2025. Keep it short": "Market Impacts",
            "Itemize potential impacts from 2025 onwards if any. Keep it short.": "Future Impacts"
        }

        # Rename the columns
        analysis_dataframe.rename(columns=column_rename_mapping, inplace=True)
        # Convert DataFrame to HTML
        analysis_html = analysis_dataframe.to_html(index=False, escape=False)

        # Display the DataFrame
        # print(analysis_dataframe)
        email_text = '<br>'.join(df_today['article_info'].tolist())
        email_html = f"""
        <h3>Mountain Valley Pipeline Scraper</h3>
        <p>{email_text}</p>
        <h3>Analysis</h3>
        {analysis_html}
        """

        # send email
        def Email(html_content):
            outlook = win32.Dispatch('outlook.application')
            mail = outlook.CreateItem(0)
            mail.To = to_email
            mail.BCC = bcc_email
            mail.Subject = "Mountain Valley Pipeline Scraper"
            mail.HTMLBody = html_content  # Setting HTMLBody instead of Body
            mail.Send()

        print(df_today)
        if len(df_today) == 0:
            print("email not sent")
            pass
        else:
            Email(email_html)
            print('Email Sent!')


    keywords_mvp = ['mountain', 'valley', 'pipeline', 'mvp', 'fracking', 'appalachian', 'etrn', 'equitrans', 'mountain valley pipeline']
    search_query_mvp = 'mountain valley pipeline'

    fetch_news(search_query_mvp, keywords_mvp)
except AttributeError as e:
    print(e)
    pass
