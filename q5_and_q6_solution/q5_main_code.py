from q5_similarity.similarity import string_similarity

s1 = input("Enter first string: ")
s2 = input("Enter second string: ")

similarity, report = string_similarity(s1, s2)
print(f"Similarity: {similarity:.2f}%")
print(report)
