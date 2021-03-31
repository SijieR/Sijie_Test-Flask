import hashlib

timestamp = '11223344'
md5_key = 'MrPupq2FkL@B$l6*7L8g24F&yZOwZKT^M7GoRS&ydKq0HiwbiY2L%MRh@05IfD'
token_local = hashlib.md5((timestamp + md5_key).encode(encoding='utf-8')).hexdigest()

a = token_local
print(a)