
from bs4 import BeautifulSoup

import urllib.parse
import pandas as pd
import csv, os, re
import requests


MARKET_CODE_DICT = {
    'kospi': 'stockMkt',
    'kosdaq': 'kosdaqMkt',
    'konex': 'konexMkt'
}


DOWNLOAD_URL = 'kind.krx.co.kr/corpgeneral/corpList.do'


def download_stock_symbols(market=None, delisted=False):
    params = {'method': 'download'}

    if market.lower() in MARKET_CODE_DICT:
        params['marketType'] = MARKET_CODE_DICT[market]

    if not delisted:
        params['searchType'] = 13

    params_string = urllib.parse.urlencode(params)
    request_url = urllib.parse.urlunsplit(['http', DOWNLOAD_URL, '', params_string, ''])

    df = pd.read_html(request_url, header=0)[0]
    df.종목코드 = df.종목코드.map('{:06d}'.format)

    return df


KOSPI200_BaseUrl = 'https://finance.naver.com/sise/entryJongmok.nhn?&page='
KOSPI200_FileName = 'KOSPI200.csv'


def download_kospi200_symbols():
    if os.path.exists(KOSPI200_FileName):
        os.remove(KOSPI200_FileName)

    for i in range(1, 22):
        try:
            url = KOSPI200_BaseUrl + str(i)
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'lxml')
            items = soup.find_all('td', {'class':'ctg'})

            stocks = []
            for item in items:
                txt = item.a.get('href')
                k = re.search('[\d]+', txt)
                if k:
                    code = k.group()
                    code = '{:06d}'.format(int(code))
                    name = item.text
                    data = code, name
                    stocks.append(data)

            with open(KOSPI200_FileName, 'a', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerows(stocks)
        except:
            pass
        finally:
            temp_for_sort = []
            with open(KOSPI200_FileName, 'r') as in_file:
                for sort_line in in_file:
                    temp_for_sort.append(sort_line)

            with open(KOSPI200_FileName, 'w') as out_file:
                seen = set()
                for line in temp_for_sort:
                    if line in seen:
                        print('%s duplicated!' % line)
                        continue  # Skip duplicated

                    seen.add(line)
                    out_file.write(line)


def test_download_stock_symbols():
    df_kospi = download_stock_symbols('kospi')
    df_codes = df_kospi[['종목코드', '회사명']]

    # Use 'ms949' encoding to prevent Korean characters being broken in the exported csv file.
    df_codes.to_csv('kospi_stocks.csv', encoding='ms949')


def test_download_kospi200_symbols():
    print("Starting download stock symbols in KOSPI200...")
    download_kospi200_symbols()
    print('Downloading completed, exported to %s' % KOSPI200_FileName)


if __name__ == '__main__':
    test_download_kospi200_symbols()

