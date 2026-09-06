from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


text = """
# Creating a Student class
class Student:

    # Constructor
    def __init__(self, name, age, course):
        self.name = name
        self.age = age
        self.course = course

    # Method
    def display_details(self):
        print("Student Name:", self.name)
        print("Age:", self.age)
        print("Course:", self.course)


# Creating objects of Student class
student1 = Student("Rahul", 21, "Computer Science")
student2 = Student("Priya", 22, "Data Science")


# Calling method using objects
student1.display_details()

print()

student2.display_details()
"""

text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=150,
    chunk_overlap=0,
    language=Language.PYTHON
)

result = text_splitter.split_text(text)

print(result)