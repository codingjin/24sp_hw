def text_to_binary(text, encoding='utf-8'):
    return ' '.join(format(byte, '08b') for char in text for byte in char.encode(encoding))

message = "新年快乐"
binary_code = text_to_binary(message)
print(binary_code)

def binary_to_text(binary_string):
    byte_array = bytearray(int(b, 2) for b in binary_string.split())
    return byte_array.decode('utf-8')

binary_code = "11100110 10010110 10110000 11100101 10111001 10110100 11100101 10111111 10101011 11100100 10111001 10010000"
text = binary_to_text(binary_code)
print(text)

binary_code = "11100101 10110000 10001111 11100101 10100100 10001111 11100110 10010110 10110000 11100101 10111001 10110100 11100101 10111111 10101011 11100100 10111001 10010000"


text = binary_to_text(binary_code)
print(text)