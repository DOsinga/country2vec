#!/usr/bin/env python3
import argparse
import psycopg2
import psycopg2.extras
import numpy as np
from stemming import porter2
import re
import string

def setup_db(connection_string):
  conn = psycopg2.connect(connection_string)
  cursor = conn.cursor()
  cursor.execute('DROP TABLE IF EXISTS word2vec')
  cursor.execute('CREATE TABLE word2vec ('
                 '    word TEXT PRIMARY KEY,'
                 '    rank INT,'
                 '    vec FLOAT[] NOT NULL DEFAULT \'{}\''
                 ')')
  cursor.execute('CREATE INDEX word2vec_vec ON word2vec USING gin(vec)')
  cursor.execute('CREATE INDEX word2vec_rank ON word2vec(rank)')
  cursor.execute('CREATE INDEX word2vec_word_pattern ON word2vec USING btree(lower(word) text_pattern_ops)')

  return conn, cursor


def read_record(binary_len, w2v):
  word = []
  while True:
    ch = w2v.read(1)
    if ch == b' ':
      break
    if ch == b'':
      raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
      word.append(ch)
  word = (b''.join(word)).decode('utf-8')
  buffer = w2v.read(binary_len)
  weights = np.frombuffer(buffer, dtype=np.float32)
  return word, weights


def main(postgres_cursor, word2vec_model, top_words, white_list):
  inserted = 0
  stems_seen = set()
  # Read the word2vec model binarily - based on word2vec from GenSim
  with open(word2vec_model, 'rb') as w2v:
    header = w2v.readline().decode('utf-8').strip()
    word_count, dimensions = map(int, header.split(' '))
    binary_len = np.dtype(np.float32).itemsize * dimensions
    rank_cut_off = 0
    stem_skipped = 0
    rank = 0
    while True:
      word, weights = read_record(binary_len, w2v)

      rank += 1
      if rank % 1000 == 0:
        print(rank, stem_skipped, rank_cut_off, len(white_list))
      if rank_cut_off and rank > rank_cut_off:
        break

      if word.lower().replace('_', ' ') in white_list:
        white_list.remove(word.lower().replace('_', ' '))
        if word.upper() == word:
          word = '_'.join(x.lower().capitalize() for x in word.split('_'))
      elif inserted > top_words:
          continue
      else:
        stem = porter2.stem(word)
        if stem in stems_seen:
          stem_skipped += 1
          continue
        stems_seen.add(stem)

      v_len = np.linalg.norm(weights)
      postgres_cursor.execute('INSERT INTO word2vec (word, rank, vec) VALUES (%s, %s, %s)',
                              (word, rank, [float(weight) / v_len for weight in weights]))
      inserted += 1
      if inserted == top_words:
        # If we run with a white list, use a grace period
        if white_list:
          rank_cut_off = rank * 6
        else:
          break
  print('left-over', ', '.join(white_list))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Download and import the necesary data')
  parser.add_argument('--postgres', type=str,
                      help='postgres connection string')
  parser.add_argument('--top_words', type=int,
                      default=30000,
                      help='Number of top terms to import into postgres')
  parser.add_argument('--white_list', type=str,
                      default='',
                      help='Try to find these words even if they are not among the top words. Searches in top_words x 4')
  parser.add_argument('--word2vec_model', type=str,
                      default='data/GoogleNews-vectors-negative300.bin',
                      help='path to pretrained word2vec model')
  args = parser.parse_args()

  postgres_conn, postgres_cursor = setup_db(args.postgres)

  main(postgres_cursor, args.word2vec_model, args.top_words, set(args.white_list.split(',')))

  postgres_conn.commit()
  postgres_cursor.close()
  postgres_conn.close()
