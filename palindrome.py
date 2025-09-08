def is_palindrome(s):
    # Remove spaces and convert to lowercase for uniformity
    s = s.replace(" ", "").lower()
    # Check if string is equal to its reverse
    return s == s[::-1]

# Input from user
word = input("Enter a word or phrase: ")

if is_palindrome(word):
    print(f"'{word}' is a palindrome!")
else:
    print(f"'{word}' is not a palindrome.")
