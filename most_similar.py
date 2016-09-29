#!/usr/bin/env python3
import argparse
import psycopg2
import psycopg2.extras
import numpy as np
import math

def main(postgres_cursor, positive, negative):
  summed = None
  both = tuple(positive + negative)
  postgres_cursor.execute('SELECT * FROM word2vec WHERE word IN %s', (both,))
  for rec in postgres_cursor:
    sign = 1 if rec['word'] in positive else -1
    summed = [x * sign + (summed[i] if summed else 0) for i, x in enumerate(rec['vec'])]
  length = 1.0 * math.sqrt(sum(val * val for val in summed))
  unit_vec = [x / length for x in summed]
  postgres_cursor.execute('SELECT word, cube_distance(cube(vec), cube(%s)) as distance from word2vec WHERE NOT word in %s order by distance limit 5', (unit_vec, both))
  for rec in postgres_cursor:
    print(rec['word'], rec['distance'])


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Download and import the necesary data')
  parser.add_argument('--postgres', type=str,
                      help='postgres connection string')
  parser.add_argument('--positive', type=str,
                      help='Positive examples separated by comma')
  parser.add_argument('--negative', type=str,
                      default='',
                      help='Negative examples separated by comma')
  args = parser.parse_args()

  conn = psycopg2.connect(args.postgres, cursor_factory=psycopg2.extras.DictCursor)
  cursor = conn.cursor()

  main(cursor, args.positive.split(','), args.negative.split(','))

