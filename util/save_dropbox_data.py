import argparse
import httplib2
import os
import pdb
import pprint
import re
import sys
import urllib2

from bs4 import BeautifulSoup

def download_data(data_url):
  '''Download and return the downloaded data.
  data_url: URL for the data to be downloaded.
  '''
  assert data_url[-4:] == 'dl=1', 'Incorrect link {}. dl=1 not found'.format(
      data_url)

  print('Will download data from: {}'.format(data_url))
  data = None
  try:
    u = urllib2.urlopen(data_url)
    data = u.read()
    u.close()
    print('Did download data from: {}'.format(data_url))
  except urllib2.HTTPError as e:
    print('Error to download data: {}'.format(e.code))
    data = None
  
  return data
 
def parse_dropbox_download_links(url):
  ''' Parse the actual URL to filter all the data URLS that should be used to
  download the data. Overwrite this function for custom parsing.'''

  http = httplib2.Http()
  status, response = http.request(url)

  soup = BeautifulSoup(response, 'html5lib')
  links = []

  pattern = re.compile('/features_raw/')
  script_text = soup.find_all('script', text=pattern)
  assert len(script_text) == 1, 'Only 1 script should contain URLs'

  script_text = script_text[0].text
  url_iter = re.finditer('https://www.dropbox.com/sh/', script_text)
  links = [script_text[s.start():s.start()+150] for s in url_iter]
  # Get the actual link 
  # https://www.dropbox.com/sh/2qk44dglzb8vasy/AAAxSWH672nv40VEFCeh3wbEa/-
  #     features_raw/VIRAT_S_040104_01_5650_6050_0_features.yml?dl=0", "ownerId
  links = [l.split('dl=0')[0] for l in links]

  # Append dl=1 to allow it to be downloaded
  links = [''.join([l, 'dl=1']) for l in links]

  # Filter links with 'VIRAT_S' in it
  links = [l for l in links if 'VIRAT_S' in l]

  return links

def extract_file_name(url):
  '''Extract the file name from URL.
  url: URL to extract filename e.g.
    https://www.dropbox.com/sh/*/*/featu-
      res_raw/VIRAT_S_040104_01_5650_6050_0_features.yml?dl=1
  '''
  suffix = url.split('VIRAT_S_')[-1]
  name = ''.join(['VIRAT_S_', suffix.split('?')[0]])
  return name

def main(url, save_dir):
  download_links = parse_dropbox_download_links(url)
  
  for i, link in enumerate(download_links): 
    data = download_data(link)

    if data is not None:
      filename = extract_file_name(link)
      file_path = os.path.join(save_dir, filename)

      with open(file_path, "wb") as f :
        f.write(data)
        print("Did write data to {}".format(file_path))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Download multiple links in Dropbox folder')
  parser.add_argument('--url', nargs='?', type=str, const=1, required=True,
      help='URL to download multiple zips from')
  parser.add_argument('--save_dir', nargs='?', type=str, const=1, required=True,
      help='Directory to save the downloaded data to')
  args = parser.parse_args()

  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    print('Did create directory {}'.format(args.save_dir))

  main(args.url, args.save_dir)

