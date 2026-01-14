"""
Python Review Dashboard - A comprehensive Dash interface for Python examples.

This application provides interactive examples covering fundamental Python concepts
including loops, conditionals, classes, exceptions, unit testing, and more.
"""

import dash
from dash import Dash, html, dcc, callback, Output, Input, State, ALL, ctx, no_update
import dash_bootstrap_components as dbc

# =============================================================================
# PYTHON EXAMPLES DATA STRUCTURE
# =============================================================================

PYTHON_EXAMPLES: dict[str, dict[str, list[dict[str, str]]]] = {
    "Variables & Data Types": {
        "description": "Python's basic data types and variable assignments.",
        "examples": [
            {
                "title": "Basic Variable Assignment",
                "code": '''# Integer
age: int = 25
print(f"Age: {age}, Type: {type(age)}")

# Float
temperature: float = 98.6
print(f"Temperature: {temperature}, Type: {type(temperature)}")

# String
name: str = "Alice"
print(f"Name: {name}, Type: {type(name)}")

# Boolean
is_student: bool = True
print(f"Is Student: {is_student}, Type: {type(is_student)}")''',
                "explanation": "Python supports dynamic typing but also allows type hints for clarity. Variables don't need explicit type declarations."
            },
            {
                "title": "Collections",
                "code": '''# List - mutable, ordered
fruits: list[str] = ["apple", "banana", "cherry"]
fruits.append("date")
print(f"List: {fruits}")

# Tuple - immutable, ordered
coordinates: tuple[float, float] = (10.5, 20.3)
print(f"Tuple: {coordinates}")

# Dictionary - key-value pairs
person: dict[str, any] = {
    "name": "Bob",
    "age": 30,
    "city": "New York"
}
print(f"Dictionary: {person}")

# Set - unique elements, unordered
unique_numbers: set[int] = {1, 2, 3, 2, 1}
print(f"Set: {unique_numbers}")  # {1, 2, 3}''',
                "explanation": "Python provides built-in collection types: lists (mutable sequences), tuples (immutable sequences), dictionaries (key-value mappings), and sets (unique elements)."
            },
            {
                "title": "Type Conversion",
                "code": '''# String to Integer
num_str: str = "42"
num_int: int = int(num_str)
print(f"String '{num_str}' to int: {num_int}")

# Integer to Float
integer_val: int = 10
float_val: float = float(integer_val)
print(f"Int {integer_val} to float: {float_val}")

# List to Tuple and vice versa
my_list: list[int] = [1, 2, 3]
my_tuple: tuple[int, ...] = tuple(my_list)
back_to_list: list[int] = list(my_tuple)
print(f"List -> Tuple -> List: {my_list} -> {my_tuple} -> {back_to_list}")''',
                "explanation": "Python provides built-in functions for converting between types: int(), float(), str(), list(), tuple(), set(), dict()."
            }
        ]
    },
    "Conditionals": {
        "description": "Control flow with if, elif, and else statements.",
        "examples": [
            {
                "title": "Basic If-Else",
                "code": '''def check_age(age: int) -> str:
    """
    Determine age category based on numeric age.
    
    Args:
        age: The person's age in years
        
    Returns:
        A string describing the age category
    """
    if age < 0:
        return "Invalid age"
    elif age < 13:
        return "Child"
    elif age < 20:
        return "Teenager"
    elif age < 65:
        return "Adult"
    else:
        return "Senior"

# Test the function
for test_age in [5, 15, 30, 70, -1]:
    print(f"Age {test_age}: {check_age(test_age)}")''',
                "explanation": "Conditionals use if, elif (else if), and else keywords. Conditions are evaluated in order, and only the first matching block executes."
            },
            {
                "title": "Ternary Operator",
                "code": '''def get_status(is_active: bool) -> str:
    """Return status string based on boolean flag."""
    return "Active" if is_active else "Inactive"

# One-line conditional assignment
score: int = 85
grade: str = "Pass" if score >= 60 else "Fail"
print(f"Score: {score}, Grade: {grade}")

# Nested ternary (use sparingly for readability)
value: int = 0
sign: str = "positive" if value > 0 else ("negative" if value < 0 else "zero")
print(f"Value {value} is {sign}")''',
                "explanation": "The ternary operator provides a concise way to write simple if-else statements in a single line: value_if_true if condition else value_if_false."
            },
            {
                "title": "Match Statement (Python 3.10+)",
                "code": '''def http_status(code: int) -> str:
    """
    Return HTTP status message using structural pattern matching.
    
    Args:
        code: HTTP status code
        
    Returns:
        Human-readable status message
    """
    match code:
        case 200:
            return "OK"
        case 201:
            return "Created"
        case 400:
            return "Bad Request"
        case 404:
            return "Not Found"
        case 500:
            return "Internal Server Error"
        case _:
            return f"Unknown status code: {code}"

# Test various status codes
for status in [200, 404, 500, 999]:
    print(f"HTTP {status}: {http_status(status)}")''',
                "explanation": "Python 3.10 introduced structural pattern matching (match-case), similar to switch statements in other languages but more powerful with pattern matching capabilities."
            }
        ]
    },
    "Loops": {
        "description": "Iteration with for and while loops.",
        "examples": [
            {
                "title": "For Loop Basics",
                "code": '''# Iterating over a list
colors: list[str] = ["red", "green", "blue"]
for color in colors:
    print(f"Color: {color}")

print("---")

# Using range()
for i in range(5):
    print(f"Index: {i}")

print("---")

# range with start, stop, step
for i in range(0, 10, 2):
    print(f"Even index: {i}")''',
                "explanation": "For loops iterate over sequences (lists, tuples, strings, ranges). The range() function generates number sequences."
            },
            {
                "title": "While Loop",
                "code": '''def countdown(n: int) -> None:
    """Print countdown from n to 1."""
    while n > 0:
        print(f"T-minus {n}")
        n -= 1
    print("Liftoff!")

countdown(5)

print("---")

# While with break and continue
def find_first_even(numbers: list[int]) -> int | None:
    """Find and return the first even number."""
    index: int = 0
    while index < len(numbers):
        if numbers[index] % 2 == 0:
            return numbers[index]
        index += 1
    return None

result = find_first_even([1, 3, 5, 8, 9])
print(f"First even: {result}")''',
                "explanation": "While loops continue executing as long as the condition is True. Use break to exit early and continue to skip to the next iteration."
            },
            {
                "title": "Loop Control & Enumerate",
                "code": '''# enumerate() for index and value
fruits: list[str] = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

print("---")

# zip() to iterate multiple sequences
names: list[str] = ["Alice", "Bob", "Charlie"]
scores: list[int] = [85, 92, 78]
for name, score in zip(names, scores):
    print(f"{name}: {score}")

print("---")

# break and continue
for i in range(10):
    if i == 3:
        continue  # Skip 3
    if i == 7:
        break  # Stop at 7
    print(i, end=" ")
print()  # Newline

# else clause (executes if loop completes without break)
for n in range(2, 5):
    for x in range(2, n):
        if n % x == 0:
            print(f"{n} = {x} * {n//x}")
            break
    else:
        print(f"{n} is prime")''',
                "explanation": "enumerate() provides index-value pairs. zip() combines multiple iterables. Loops can have an else clause that runs if no break occurred."
            }
        ]
    },
    "Functions": {
        "description": "Defining and using functions with various parameter types.",
        "examples": [
            {
                "title": "Basic Functions",
                "code": '''def greet(name: str) -> str:
    """
    Generate a greeting message.
    
    Args:
        name: The name to greet
        
    Returns:
        A greeting string
    """
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

print(greet("World"))
print(f"5 + 3 = {add(5, 3)}")''',
                "explanation": "Functions are defined with def, can have type hints, and should include docstrings explaining their purpose."
            },
            {
                "title": "Default & Keyword Arguments",
                "code": '''def create_profile(
    name: str,
    age: int,
    city: str = "Unknown",
    occupation: str = "Not specified"
) -> dict[str, any]:
    """
    Create a user profile dictionary.
    
    Args:
        name: User's name (required)
        age: User's age (required)
        city: User's city (optional, default "Unknown")
        occupation: User's job (optional)
        
    Returns:
        Dictionary containing user profile
    """
    return {
        "name": name,
        "age": age,
        "city": city,
        "occupation": occupation
    }

# Positional arguments
print(create_profile("Alice", 25))

# Keyword arguments
print(create_profile(name="Bob", age=30, city="NYC"))

# Mixed
print(create_profile("Charlie", 35, occupation="Engineer"))''',
                "explanation": "Parameters with default values are optional. Keyword arguments can be passed in any order and improve code readability."
            },
            {
                "title": "*args and **kwargs",
                "code": '''def sum_all(*args: int) -> int:
    """
    Sum any number of integer arguments.
    
    Args:
        *args: Variable number of integers
        
    Returns:
        Sum of all arguments
    """
    return sum(args)

def print_info(**kwargs: any) -> None:
    """
    Print all keyword arguments.
    
    Args:
        **kwargs: Arbitrary keyword arguments
    """
    for key, value in kwargs.items():
        print(f"{key}: {value}")

def combined(*args: int, **kwargs: str) -> None:
    """Demonstrate combining *args and **kwargs."""
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

print(f"Sum: {sum_all(1, 2, 3, 4, 5)}")
print("---")
print_info(name="Alice", age=25, city="Boston")
print("---")
combined(1, 2, 3, greeting="Hello", target="World")''',
                "explanation": "*args collects positional arguments into a tuple. **kwargs collects keyword arguments into a dictionary. They enable flexible function signatures."
            },
            {
                "title": "Lambda Functions",
                "code": '''# Simple lambda
square = lambda x: x ** 2
print(f"Square of 5: {square(5)}")

# Lambda with multiple arguments
multiply = lambda x, y: x * y
print(f"3 * 4 = {multiply(3, 4)}")

# Common use: sorting
students: list[dict[str, any]] = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78}
]

# Sort by grade
sorted_students = sorted(students, key=lambda s: s["grade"], reverse=True)
for student in sorted_students:
    print(f"{student['name']}: {student['grade']}")

# With filter and map
numbers: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens: list[int] = list(filter(lambda x: x % 2 == 0, numbers))
squared: list[int] = list(map(lambda x: x ** 2, evens))
print(f"Even numbers squared: {squared}")''',
                "explanation": "Lambda functions are anonymous, single-expression functions. They're useful for short operations, especially with map(), filter(), and sorted()."
            }
        ]
    },
    "Classes & OOP": {
        "description": "Object-Oriented Programming with classes.",
        "examples": [
            {
                "title": "Basic Class",
                "code": '''class Dog:
    """A simple Dog class demonstrating basic OOP concepts."""
    
    # Class attribute (shared by all instances)
    species: str = "Canis familiaris"
    
    def __init__(self, name: str, age: int) -> None:
        """
        Initialize a Dog instance.
        
        Args:
            name: The dog's name
            age: The dog's age in years
        """
        # Instance attributes
        self.name = name
        self.age = age
    
    def bark(self) -> str:
        """Return the dog's bark."""
        return f"{self.name} says Woof!"
    
    def describe(self) -> str:
        """Return a description of the dog."""
        return f"{self.name} is {self.age} years old"
    
    def __str__(self) -> str:
        """String representation for print()."""
        return f"Dog({self.name}, {self.age})"
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Dog(name='{self.name}', age={self.age})"

# Create instances
buddy = Dog("Buddy", 3)
max_dog = Dog("Max", 5)

print(buddy.bark())
print(max_dog.describe())
print(f"Species: {Dog.species}")
print(f"Repr: {repr(buddy)}")''',
                "explanation": "Classes bundle data (attributes) and behavior (methods). __init__ is the constructor. __str__ and __repr__ provide string representations."
            },
            {
                "title": "Inheritance",
                "code": '''class Animal:
    """Base class for animals."""
    
    def __init__(self, name: str) -> None:
        self.name = name
    
    def speak(self) -> str:
        """Override in subclasses."""
        raise NotImplementedError("Subclass must implement speak()")
    
    def describe(self) -> str:
        return f"I am {self.name}"

class Cat(Animal):
    """Cat class inheriting from Animal."""
    
    def __init__(self, name: str, indoor: bool = True) -> None:
        super().__init__(name)  # Call parent constructor
        self.indoor = indoor
    
    def speak(self) -> str:
        return f"{self.name} says Meow!"
    
    def describe(self) -> str:
        base = super().describe()
        status = "indoor" if self.indoor else "outdoor"
        return f"{base}, an {status} cat"

class Bird(Animal):
    """Bird class inheriting from Animal."""
    
    def __init__(self, name: str, can_fly: bool = True) -> None:
        super().__init__(name)
        self.can_fly = can_fly
    
    def speak(self) -> str:
        return f"{self.name} says Tweet!"

# Polymorphism in action
animals: list[Animal] = [
    Cat("Whiskers"),
    Bird("Tweety"),
    Cat("Shadow", indoor=False)
]

for animal in animals:
    print(f"{animal.speak()} - {animal.describe()}")''',
                "explanation": "Inheritance allows classes to inherit attributes and methods from parent classes. super() calls parent methods. Polymorphism lets different classes share the same interface."
            },
            {
                "title": "Properties and Encapsulation",
                "code": '''class BankAccount:
    """
    Bank account with encapsulated balance.
    
    Demonstrates property decorators for controlled access.
    """
    
    def __init__(self, owner: str, initial_balance: float = 0) -> None:
        self.owner = owner
        self._balance = initial_balance  # Convention: _ prefix = "protected"
    
    @property
    def balance(self) -> float:
        """Get the current balance (read-only property)."""
        return self._balance
    
    @property
    def is_overdrawn(self) -> bool:
        """Check if account is overdrawn."""
        return self._balance < 0
    
    def deposit(self, amount: float) -> None:
        """
        Deposit money into the account.
        
        Args:
            amount: Amount to deposit (must be positive)
            
        Raises:
            ValueError: If amount is not positive
        """
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
        print(f"Deposited ${amount:.2f}. New balance: ${self._balance:.2f}")
    
    def withdraw(self, amount: float) -> bool:
        """
        Withdraw money from the account.
        
        Args:
            amount: Amount to withdraw
            
        Returns:
            True if withdrawal successful, False otherwise
        """
        if amount <= 0:
            print("Withdrawal amount must be positive")
            return False
        if amount > self._balance:
            print(f"Insufficient funds. Balance: ${self._balance:.2f}")
            return False
        self._balance -= amount
        print(f"Withdrew ${amount:.2f}. New balance: ${self._balance:.2f}")
        return True

# Usage
account = BankAccount("Alice", 100)
print(f"Owner: {account.owner}, Balance: ${account.balance:.2f}")
account.deposit(50)
account.withdraw(30)
account.withdraw(200)  # Will fail
print(f"Is overdrawn: {account.is_overdrawn}")''',
                "explanation": "@property decorator creates getter methods that look like attribute access. This enables validation, computed properties, and encapsulation while maintaining a clean API."
            },
            {
                "title": "Class Methods and Static Methods",
                "code": '''from datetime import date

class Employee:
    """Employee class demonstrating class and static methods."""
    
    raise_percentage: float = 1.05  # 5% raise
    employee_count: int = 0
    
    def __init__(self, name: str, salary: float, birth_year: int) -> None:
        self.name = name
        self.salary = salary
        self.birth_year = birth_year
        Employee.employee_count += 1
    
    def apply_raise(self) -> None:
        """Apply the standard raise to this employee."""
        self.salary *= self.raise_percentage
    
    @classmethod
    def set_raise_percentage(cls, percentage: float) -> None:
        """
        Set the raise percentage for all employees.
        
        Args:
            percentage: New raise multiplier (e.g., 1.10 for 10%)
        """
        cls.raise_percentage = percentage
    
    @classmethod
    def from_string(cls, emp_string: str) -> "Employee":
        """
        Create Employee from hyphen-separated string.
        
        Args:
            emp_string: Format "name-salary-birth_year"
            
        Returns:
            New Employee instance
        """
        name, salary, birth_year = emp_string.split("-")
        return cls(name, float(salary), int(birth_year))
    
    @staticmethod
    def is_workday(day: date) -> bool:
        """
        Check if a date is a workday.
        
        Args:
            day: Date to check
            
        Returns:
            True if Monday-Friday, False otherwise
        """
        return day.weekday() < 5  # 0-4 are Mon-Fri

# Regular instantiation
emp1 = Employee("Alice", 50000, 1990)

# Using class method as alternative constructor
emp2 = Employee.from_string("Bob-60000-1985")

# Using class method to modify class attribute
Employee.set_raise_percentage(1.10)
emp1.apply_raise()
print(f"{emp1.name}'s new salary: ${emp1.salary:.2f}")

# Static method
today = date.today()
print(f"Is today a workday? {Employee.is_workday(today)}")
print(f"Total employees: {Employee.employee_count}")''',
                "explanation": "@classmethod receives the class as first argument (cls) - useful for alternative constructors. @staticmethod doesn't receive class or instance - used for utility functions related to the class."
            }
        ]
    },
    "Exceptions": {
        "description": "Error handling with try-except blocks.",
        "examples": [
            {
                "title": "Basic Exception Handling",
                "code": '''def safe_divide(a: float, b: float) -> float | None:
    """
    Safely divide two numbers with error handling.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        Result of division or None if error
    """
    try:
        result = a / b
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    except TypeError as e:
        print(f"Error: Invalid types - {e}")
        return None
    else:
        # Runs if no exception occurred
        print(f"Division successful: {a} / {b} = {result}")
        return result
    finally:
        # Always runs
        print("Division operation complete")

# Test cases
safe_divide(10, 2)
print("---")
safe_divide(10, 0)
print("---")
safe_divide("10", 2)''',
                "explanation": "try-except handles errors gracefully. else runs if no exception occurred. finally always runs (useful for cleanup). Catch specific exceptions for targeted handling."
            },
            {
                "title": "Multiple Exceptions",
                "code": '''def process_data(data: dict, key: str) -> int:
    """
    Process data from dictionary with comprehensive error handling.
    
    Args:
        data: Dictionary containing data
        key: Key to look up
        
    Returns:
        Processed integer value
        
    Raises:
        Various exceptions based on error type
    """
    try:
        value = data[key]
        result = int(value) * 2
        return result
    except KeyError:
        print(f"Key '{key}' not found in data")
        raise
    except ValueError:
        print(f"Cannot convert '{data[key]}' to integer")
        raise
    except (TypeError, AttributeError) as e:
        print(f"Data structure error: {e}")
        raise

# Test cases
test_data = {"a": "10", "b": "hello", "c": None}

# Success case
try:
    print(f"Result: {process_data(test_data, 'a')}")
except Exception:
    pass

# KeyError
try:
    process_data(test_data, "missing")
except KeyError:
    print("Caught KeyError")

# ValueError
try:
    process_data(test_data, "b")
except ValueError:
    print("Caught ValueError")''',
                "explanation": "Multiple exception types can be caught separately or grouped with parentheses. 'raise' re-raises the current exception. Exception chaining preserves the original traceback."
            },
            {
                "title": "Custom Exceptions",
                "code": '''class ValidationError(Exception):
    """Custom exception for validation failures."""
    
    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

class InsufficientFundsError(Exception):
    """Exception raised when account has insufficient funds."""
    
    def __init__(self, balance: float, amount: float) -> None:
        self.balance = balance
        self.amount = amount
        self.deficit = amount - balance
        super().__init__(
            f"Cannot withdraw ${amount:.2f}. "
            f"Balance: ${balance:.2f}, Deficit: ${self.deficit:.2f}"
        )

def validate_user(name: str, age: int) -> None:
    """
    Validate user data.
    
    Raises:
        ValidationError: If validation fails
    """
    if not name or not name.strip():
        raise ValidationError("name", "Name cannot be empty")
    if age < 0:
        raise ValidationError("age", "Age cannot be negative")
    if age > 150:
        raise ValidationError("age", "Age seems unrealistic")
    print(f"User '{name}' (age {age}) validated successfully")

# Test validation
for name, age in [("Alice", 25), ("", 30), ("Bob", -5)]:
    try:
        validate_user(name, age)
    except ValidationError as e:
        print(f"Validation failed - {e.field}: {e.message}")''',
                "explanation": "Custom exceptions inherit from Exception. They can store additional context about the error, making debugging and error handling more informative."
            }
        ]
    },
    "List Comprehensions": {
        "description": "Concise syntax for creating lists, dicts, and sets.",
        "examples": [
            {
                "title": "Basic List Comprehensions",
                "code": '''# Simple transformation
numbers: list[int] = [1, 2, 3, 4, 5]
squares: list[int] = [x ** 2 for x in numbers]
print(f"Squares: {squares}")

# With condition (filter)
evens: list[int] = [x for x in range(10) if x % 2 == 0]
print(f"Evens: {evens}")

# String manipulation
words: list[str] = ["hello", "world", "python"]
upper_words: list[str] = [word.upper() for word in words]
print(f"Uppercase: {upper_words}")

# Conditional expression (ternary in comprehension)
labels: list[str] = ["even" if x % 2 == 0 else "odd" for x in range(5)]
print(f"Labels: {labels}")''',
                "explanation": "List comprehensions provide a concise way to create lists. Syntax: [expression for item in iterable if condition]. They're often faster and more readable than loops."
            },
            {
                "title": "Nested Comprehensions",
                "code": '''# Flatten a 2D list
matrix: list[list[int]] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened: list[int] = [num for row in matrix for num in row]
print(f"Flattened: {flattened}")

# Create a 2D list (matrix)
size: int = 3
identity: list[list[int]] = [
    [1 if i == j else 0 for j in range(size)]
    for i in range(size)
]
print("Identity matrix:")
for row in identity:
    print(row)

# Cartesian product
colors: list[str] = ["red", "blue"]
sizes: list[str] = ["S", "M", "L"]
combinations: list[tuple[str, str]] = [
    (color, size) for color in colors for size in sizes
]
print(f"Combinations: {combinations}")''',
                "explanation": "Nested comprehensions can iterate over multiple sequences. The outer loop comes first in the syntax. Use sparingly - complex nesting hurts readability."
            },
            {
                "title": "Dict and Set Comprehensions",
                "code": '''# Dictionary comprehension
names: list[str] = ["alice", "bob", "charlie"]
name_lengths: dict[str, int] = {name: len(name) for name in names}
print(f"Name lengths: {name_lengths}")

# Dict from two lists
keys: list[str] = ["a", "b", "c"]
values: list[int] = [1, 2, 3]
combined: dict[str, int] = {k: v for k, v in zip(keys, values)}
print(f"Combined dict: {combined}")

# Filtering dict
scores: dict[str, int] = {"Alice": 85, "Bob": 72, "Charlie": 91, "Diana": 68}
passing: dict[str, int] = {name: score for name, score in scores.items() if score >= 75}
print(f"Passing students: {passing}")

# Set comprehension
numbers: list[int] = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique_squares: set[int] = {x ** 2 for x in numbers}
print(f"Unique squares: {unique_squares}")

# Dict inversion
original: dict[str, int] = {"a": 1, "b": 2, "c": 3}
inverted: dict[int, str] = {v: k for k, v in original.items()}
print(f"Inverted: {inverted}")''',
                "explanation": "Dict comprehensions use {key: value for ...}. Set comprehensions use {value for ...}. Both support filtering and transformations like list comprehensions."
            }
        ]
    },
    "File I/O": {
        "description": "Reading from and writing to files.",
        "examples": [
            {
                "title": "Reading Files",
                "code": '''from pathlib import Path

# Using context manager (recommended)
def read_file_safely(filepath: str) -> str | None:
    """
    Safely read entire file contents.
    
    Args:
        filepath: Path to file
        
    Returns:
        File contents or None if error
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except PermissionError:
        print(f"Permission denied: {filepath}")
        return None

# Reading line by line (memory efficient for large files)
def count_lines(filepath: str) -> int:
    """Count lines in a file without loading it all into memory."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            count += 1
    return count

# Using pathlib (modern approach)
def read_with_pathlib(filepath: str) -> str:
    """Read file using pathlib."""
    path = Path(filepath)
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"File not found: {filepath}")

# Example usage (simulated)
print("File reading examples demonstrated above")
print("Use 'with open()' for automatic resource management")''',
                "explanation": "Always use 'with' statements (context managers) for file operations - they ensure files are properly closed. Use encoding='utf-8' for text files."
            },
            {
                "title": "Writing Files",
                "code": '''from pathlib import Path
import json

def write_text_file(filepath: str, content: str) -> bool:
    """
    Write text content to file.
    
    Args:
        filepath: Destination path
        content: Text to write
        
    Returns:
        True if successful
    """
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Successfully wrote to {filepath}")
        return True
    except IOError as e:
        print(f"Error writing file: {e}")
        return False

def append_to_file(filepath: str, content: str) -> None:
    """Append content to existing file."""
    with open(filepath, "a", encoding="utf-8") as file:
        file.write(content + "\\n")

def write_lines(filepath: str, lines: list[str]) -> None:
    """Write multiple lines to file."""
    with open(filepath, "w", encoding="utf-8") as file:
        file.writelines(line + "\\n" for line in lines)

def save_json(filepath: str, data: dict) -> None:
    """Save dictionary as JSON file."""
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    print(f"JSON saved to {filepath}")

# Examples (not actually executed to avoid file creation)
print("File writing modes:")
print("  'w' - write (overwrites existing)")
print("  'a' - append (adds to existing)")
print("  'x' - exclusive create (fails if exists)")''',
                "explanation": "Write modes: 'w' overwrites, 'a' appends, 'x' creates exclusively. writelines() writes an iterable. json.dump() serializes Python objects to JSON."
            },
            {
                "title": "Working with Paths",
                "code": '''from pathlib import Path

# Create Path objects
current_dir = Path(".")
home_dir = Path.home()
data_file = Path("data") / "input.txt"  # Platform-independent path joining

print(f"Current directory: {current_dir.absolute()}")
print(f"Home directory: {home_dir}")
print(f"Data file path: {data_file}")

# Path properties
example_path = Path("/home/user/documents/report.pdf")
print(f"\\nPath: {example_path}")
print(f"  Name: {example_path.name}")
print(f"  Stem: {example_path.stem}")
print(f"  Suffix: {example_path.suffix}")
print(f"  Parent: {example_path.parent}")
print(f"  Parts: {example_path.parts}")

# Common operations
def demonstrate_path_operations() -> None:
    """Demonstrate common pathlib operations."""
    path = Path("example")
    
    # Check existence
    print(f"\\nExists: {path.exists()}")
    print(f"Is file: {path.is_file()}")
    print(f"Is directory: {path.is_dir()}")
    
    # Globbing (finding files)
    print("\\nPython files in current directory:")
    for py_file in Path(".").glob("*.py"):
        print(f"  {py_file}")

demonstrate_path_operations()''',
                "explanation": "pathlib.Path provides an object-oriented interface for filesystem paths. It handles platform differences automatically and provides convenient methods for common operations."
            }
        ]
    },
    "Decorators": {
        "description": "Functions that modify other functions' behavior.",
        "examples": [
            {
                "title": "Basic Decorator",
                "code": '''from functools import wraps
from typing import Callable, Any

def timer(func: Callable) -> Callable:
    """
    Decorator that measures function execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with timing
    """
    import time
    
    @wraps(func)  # Preserves function metadata
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function(n: int) -> int:
    """Calculate sum of squares (slow version for demo)."""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

# Using the decorated function
result = slow_function(100000)
print(f"Result: {result}")''',
                "explanation": "Decorators wrap functions to extend their behavior. @wraps preserves the original function's name and docstring. The @ syntax is syntactic sugar for func = decorator(func)."
            },
            {
                "title": "Decorator with Arguments",
                "code": '''from functools import wraps
from typing import Callable, Any

def repeat(times: int) -> Callable:
    """
    Decorator factory that repeats function execution.
    
    Args:
        times: Number of times to repeat
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = None
            for i in range(times):
                print(f"Execution {i + 1}/{times}")
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

def validate_types(*types: type) -> Callable:
    """
    Decorator that validates argument types.
    
    Args:
        *types: Expected types for each positional argument
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for arg, expected_type in zip(args, types):
                if not isinstance(arg, expected_type):
                    raise TypeError(
                        f"Expected {expected_type.__name__}, "
                        f"got {type(arg).__name__}"
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(times=3)
def greet(name: str) -> None:
    """Print a greeting."""
    print(f"Hello, {name}!")

@validate_types(str, int)
def create_message(name: str, count: int) -> str:
    """Create a repeated message."""
    return f"{name} " * count

greet("World")
print("---")
print(create_message("Hi", 3))''',
                "explanation": "Decorators with arguments require an extra level of nesting: a factory function that returns the actual decorator. This pattern is called a 'decorator factory'."
            },
            {
                "title": "Class-based Decorator",
                "code": '''from typing import Callable, Any

class CallCounter:
    """
    Decorator class that counts function calls.
    
    Attributes:
        func: The wrapped function
        count: Number of times the function has been called
    """
    
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.count = 0
        # Preserve function metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the wrapped function and increment counter."""
        self.count += 1
        print(f"Call #{self.count} to {self.func.__name__}")
        return self.func(*args, **kwargs)
    
    def reset(self) -> None:
        """Reset the call counter."""
        self.count = 0

class Memoize:
    """Decorator class that caches function results."""
    
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.cache: dict[tuple, Any] = {}
    
    def __call__(self, *args: Any) -> Any:
        if args in self.cache:
            print(f"Cache hit for {args}")
            return self.cache[args]
        print(f"Computing for {args}")
        result = self.func(*args)
        self.cache[args] = result
        return result

@CallCounter
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@Memoize
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Test CallCounter
print(add(2, 3))
print(add(5, 7))
print(f"Total calls: {add.count}")

print("---")

# Test Memoize
print(f"fib(10) = {fibonacci(10)}")''',
                "explanation": "Classes can be decorators by implementing __call__. This allows decorators to maintain state (like call counts or caches) and provide additional methods."
            }
        ]
    },
    "Generators": {
        "description": "Memory-efficient iterators using yield.",
        "examples": [
            {
                "title": "Basic Generator",
                "code": '''from typing import Generator

def countdown(n: int) -> Generator[int, None, None]:
    """
    Generator that counts down from n to 1.
    
    Args:
        n: Starting number
        
    Yields:
        Numbers from n down to 1
    """
    while n > 0:
        yield n
        n -= 1

# Using the generator
print("Countdown:")
for num in countdown(5):
    print(num)

# Generator creates values on-demand
gen = countdown(3)
print(f"\\nManual iteration:")
print(f"First: {next(gen)}")
print(f"Second: {next(gen)}")
print(f"Third: {next(gen)}")
# next(gen) would raise StopIteration

# Memory comparison
def get_squares_list(n: int) -> list[int]:
    """Return list of squares (stores all in memory)."""
    return [x ** 2 for x in range(n)]

def get_squares_gen(n: int) -> Generator[int, None, None]:
    """Yield squares one at a time (memory efficient)."""
    for x in range(n):
        yield x ** 2

print(f"\\nSum of squares: {sum(get_squares_gen(1000000))}")''',
                "explanation": "Generators use 'yield' instead of 'return'. They produce values lazily, one at a time, making them memory-efficient for large sequences."
            },
            {
                "title": "Generator Expressions",
                "code": '''# Generator expression (like list comprehension but lazy)
squares_gen = (x ** 2 for x in range(10))
print(f"Generator object: {squares_gen}")
print(f"Sum: {sum(squares_gen)}")

# Memory comparison
import sys

# List comprehension - stores all values
list_comp = [x ** 2 for x in range(1000)]
print(f"\\nList size: {sys.getsizeof(list_comp)} bytes")

# Generator expression - stores only the expression
gen_exp = (x ** 2 for x in range(1000))
print(f"Generator size: {sys.getsizeof(gen_exp)} bytes")

# Chaining generators
def evens(n: int):
    """Generate even numbers up to n."""
    return (x for x in range(n) if x % 2 == 0)

def squares(nums):
    """Generate squares of input numbers."""
    return (x ** 2 for x in nums)

# Pipeline: even numbers -> squared
pipeline = squares(evens(20))
print(f"\\nEven squares: {list(pipeline)}")

# Using in functions that accept iterables
numbers = (x for x in range(1, 101))
print(f"Sum 1-100: {sum(numbers)}")
print(f"Max of squares: {max(x**2 for x in range(10))}")''',
                "explanation": "Generator expressions use parentheses instead of brackets. They're ideal for single-use iterations and can be passed directly to functions like sum(), max(), etc."
            },
            {
                "title": "Advanced Generator Patterns",
                "code": '''from typing import Generator, Iterator

def fibonacci_gen() -> Generator[int, None, None]:
    """Generate infinite Fibonacci sequence."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def take(n: int, iterable: Iterator) -> Generator:
    """Take first n items from any iterable."""
    for i, item in enumerate(iterable):
        if i >= n:
            break
        yield item

# Get first 10 Fibonacci numbers
fib_10 = list(take(10, fibonacci_gen()))
print(f"First 10 Fibonacci: {fib_10}")

def read_large_file(filepath: str) -> Generator[str, None, None]:
    """
    Memory-efficient file reader.
    
    Yields one line at a time instead of loading entire file.
    """
    with open(filepath, "r") as file:
        for line in file:
            yield line.strip()

# Generator with send() - two-way communication
def accumulator() -> Generator[int, int, None]:
    """
    Generator that accumulates values sent to it.
    
    Yields:
        Running total
    """
    total = 0
    while True:
        value = yield total
        if value is not None:
            total += value

# Using send()
acc = accumulator()
print(f"\\nAccumulator demo:")
print(f"Initial: {next(acc)}")  # Start the generator
print(f"After sending 5: {acc.send(5)}")
print(f"After sending 10: {acc.send(10)}")
print(f"After sending 3: {acc.send(3)}")''',
                "explanation": "Generators can be infinite (paired with take/islice). send() enables two-way communication. Generators are ideal for processing large files or streams."
            }
        ]
    },
    "Context Managers": {
        "description": "Managing resources with 'with' statements.",
        "examples": [
            {
                "title": "Custom Context Manager (Class)",
                "code": '''from typing import Any

class Timer:
    """Context manager that measures code block execution time."""
    
    def __init__(self, name: str = "Block") -> None:
        self.name = name
        self.start_time: float = 0
        self.end_time: float = 0
    
    def __enter__(self) -> "Timer":
        """Called when entering the 'with' block."""
        import time
        self.start_time = time.perf_counter()
        print(f"Starting {self.name}...")
        return self
    
    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any
    ) -> bool:
        """
        Called when exiting the 'with' block.
        
        Args:
            exc_type: Exception type if error occurred
            exc_val: Exception instance
            exc_tb: Traceback object
            
        Returns:
            False to propagate exceptions, True to suppress
        """
        import time
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        print(f"{self.name} completed in {elapsed:.4f} seconds")
        
        if exc_type is not None:
            print(f"Exception occurred: {exc_val}")
        
        return False  # Don't suppress exceptions
    
    @property
    def elapsed(self) -> float:
        """Return elapsed time."""
        return self.end_time - self.start_time

# Using the context manager
with Timer("Calculation") as t:
    total = sum(i ** 2 for i in range(100000))
    print(f"Result: {total}")

print(f"Recorded time: {t.elapsed:.4f}s")''',
                "explanation": "__enter__ runs at the start of the 'with' block and can return a value. __exit__ runs at the end, even if exceptions occur. Return True from __exit__ to suppress exceptions."
            },
            {
                "title": "Context Manager with contextlib",
                "code": '''from contextlib import contextmanager
from typing import Generator

@contextmanager
def managed_resource(name: str) -> Generator[dict, None, None]:
    """
    Context manager using decorator syntax.
    
    Args:
        name: Resource name for logging
        
    Yields:
        A dictionary representing the resource
    """
    # Setup (like __enter__)
    print(f"Acquiring resource: {name}")
    resource = {"name": name, "status": "active"}
    
    try:
        yield resource  # This is what 'as' receives
    except Exception as e:
        print(f"Error with resource {name}: {e}")
        resource["status"] = "error"
        raise
    finally:
        # Cleanup (like __exit__)
        print(f"Releasing resource: {name}")
        resource["status"] = "released"

# Usage
with managed_resource("database") as db:
    print(f"Using: {db}")
    db["data"] = "some value"

print("---")

@contextmanager
def temporary_change(obj: dict, key: str, new_value: Any) -> Generator[None, None, None]:
    """Temporarily change a dictionary value."""
    old_value = obj.get(key)
    obj[key] = new_value
    try:
        yield
    finally:
        if old_value is None:
            del obj[key]
        else:
            obj[key] = old_value

config = {"debug": False, "level": "INFO"}
print(f"Before: {config}")

with temporary_change(config, "debug", True):
    print(f"Inside: {config}")

print(f"After: {config}")''',
                "explanation": "@contextmanager decorator simplifies context manager creation. Code before 'yield' is __enter__, code after is __exit__. Use try/finally to ensure cleanup runs."
            },
            {
                "title": "Multiple Context Managers",
                "code": '''from contextlib import contextmanager, ExitStack
from typing import Generator

@contextmanager
def open_file_safely(path: str, mode: str = "r") -> Generator:
    """Simulated safe file opener."""
    print(f"Opening {path} in mode '{mode}'")
    file_obj = {"path": path, "mode": mode, "content": []}
    try:
        yield file_obj
    finally:
        print(f"Closing {path}")

# Multiple context managers in one 'with'
with (
    open_file_safely("input.txt", "r") as infile,
    open_file_safely("output.txt", "w") as outfile
):
    print(f"Reading from {infile['path']}")
    print(f"Writing to {outfile['path']}")

print("---")

# Dynamic number of context managers with ExitStack
@contextmanager
def numbered_resource(n: int) -> Generator[int, None, None]:
    """Simple numbered resource for demo."""
    print(f"  Acquiring resource {n}")
    try:
        yield n
    finally:
        print(f"  Releasing resource {n}")

def process_multiple_resources(count: int) -> list[int]:
    """Process a dynamic number of resources."""
    results = []
    
    with ExitStack() as stack:
        # Dynamically enter multiple context managers
        resources = [
            stack.enter_context(numbered_resource(i))
            for i in range(count)
        ]
        
        print(f"All {count} resources acquired: {resources}")
        results = [r * 2 for r in resources]
    
    return results

print("\\nExitStack demo:")
result = process_multiple_resources(3)
print(f"Results: {result}")''',
                "explanation": "Multiple context managers can be combined in one 'with' statement. ExitStack handles a dynamic number of context managers, useful when the count isn't known until runtime."
            }
        ]
    },
    "Unit Testing": {
        "description": "Writing and running tests with unittest and pytest.",
        "examples": [
            {
                "title": "unittest Basics",
                "code": '''import unittest

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

class TestMathFunctions(unittest.TestCase):
    """Test cases for math functions."""
    
    def setUp(self) -> None:
        """Set up test fixtures (runs before each test)."""
        self.test_values = [(1, 2, 3), (0, 0, 0), (-1, 1, 0)]
    
    def tearDown(self) -> None:
        """Clean up after each test."""
        pass
    
    def test_add_positive_numbers(self) -> None:
        """Test addition with positive numbers."""
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(10, 20), 30)
    
    def test_add_negative_numbers(self) -> None:
        """Test addition with negative numbers."""
        self.assertEqual(add(-1, -1), -2)
        self.assertEqual(add(-5, 3), -2)
    
    def test_add_zero(self) -> None:
        """Test addition with zero."""
        self.assertEqual(add(0, 5), 5)
        self.assertEqual(add(5, 0), 5)
    
    def test_divide_normal(self) -> None:
        """Test normal division."""
        self.assertEqual(divide(10, 2), 5)
        self.assertAlmostEqual(divide(1, 3), 0.333, places=3)
    
    def test_divide_by_zero(self) -> None:
        """Test that division by zero raises ValueError."""
        with self.assertRaises(ValueError) as context:
            divide(10, 0)
        self.assertIn("zero", str(context.exception))

# Run tests (in actual usage, use: python -m unittest)
print("unittest example - run with: python -m unittest <filename>")
print("\\nCommon assertions:")
print("  assertEqual(a, b)      - a == b")
print("  assertTrue(x)          - bool(x) is True")
print("  assertFalse(x)         - bool(x) is False")
print("  assertIsNone(x)        - x is None")
print("  assertIn(a, b)         - a in b")
print("  assertRaises(exc)      - exception is raised")
print("  assertAlmostEqual()    - for float comparison")''',
                "explanation": "unittest is Python's built-in testing framework. Test classes inherit from TestCase. setUp/tearDown run before/after each test. Use descriptive method names starting with 'test_'."
            },
            {
                "title": "pytest Style Tests",
                "code": '''import pytest
from typing import Any

# Functions to test
def greet(name: str) -> str:
    """Return greeting message."""
    if not name:
        raise ValueError("Name cannot be empty")
    return f"Hello, {name}!"

def calculate_average(numbers: list[float]) -> float:
    """Calculate average of numbers."""
    if not numbers:
        raise ValueError("List cannot be empty")
    return sum(numbers) / len(numbers)

# pytest uses simple functions (no class needed)
def test_greet_normal() -> None:
    """Test normal greeting."""
    assert greet("Alice") == "Hello, Alice!"
    assert greet("Bob") == "Hello, Bob!"

def test_greet_empty_name() -> None:
    """Test that empty name raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        greet("")

def test_average_normal() -> None:
    """Test average calculation."""
    assert calculate_average([1, 2, 3, 4, 5]) == 3.0
    assert calculate_average([10]) == 10.0

def test_average_floats() -> None:
    """Test average with float comparison."""
    result = calculate_average([1, 2, 3])
    assert result == pytest.approx(2.0)

# Parametrized tests - run same test with different inputs
@pytest.mark.parametrize("numbers,expected", [
    ([1, 2, 3], 2.0),
    ([10, 20], 15.0),
    ([5], 5.0),
    ([-1, 1], 0.0),
])
def test_average_parametrized(numbers: list[float], expected: float) -> None:
    """Test average with multiple inputs."""
    assert calculate_average(numbers) == expected

# Fixtures for shared setup
@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Provide sample data for tests."""
    return {
        "users": ["Alice", "Bob", "Charlie"],
        "scores": [85, 92, 78]
    }

def test_with_fixture(sample_data: dict[str, Any]) -> None:
    """Test using fixture data."""
    assert len(sample_data["users"]) == 3
    assert calculate_average(sample_data["scores"]) == 85.0

print("pytest example - run with: pytest <filename> -v")
print("\\npytest advantages:")
print("  - Simple assert statements")
print("  - Powerful fixtures system")
print("  - Parametrized tests")
print("  - Better error messages")
print("  - Rich plugin ecosystem")''',
                "explanation": "pytest uses plain assert statements and functions. @pytest.mark.parametrize runs the same test with different inputs. Fixtures provide reusable test data and setup."
            },
            {
                "title": "Mocking and Test Doubles",
                "code": '''from unittest.mock import Mock, patch, MagicMock
from typing import Any

# Example class that depends on external service
class UserService:
    """Service that fetches user data from an API."""
    
    def __init__(self, api_client: Any) -> None:
        self.api_client = api_client
    
    def get_user_name(self, user_id: int) -> str:
        """Fetch user name from API."""
        response = self.api_client.get(f"/users/{user_id}")
        return response["name"]
    
    def create_user(self, name: str) -> int:
        """Create new user via API."""
        response = self.api_client.post("/users", {"name": name})
        return response["id"]

# Testing with Mock objects
def test_get_user_name_with_mock() -> None:
    """Test get_user_name using a mock API client."""
    # Create mock
    mock_client = Mock()
    mock_client.get.return_value = {"name": "Alice", "id": 1}
    
    # Inject mock
    service = UserService(mock_client)
    result = service.get_user_name(1)
    
    # Assertions
    assert result == "Alice"
    mock_client.get.assert_called_once_with("/users/1")
    print(f"Test passed: got '{result}'")

# Using patch as decorator
def fetch_data(url: str) -> dict:
    """Simulated function that fetches data."""
    import requests
    return requests.get(url).json()

def test_with_patch() -> None:
    """Demonstrate patching an external module."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"data": "mocked"}
        
        # In real test, this would call the patched function
        print("Patch allows replacing external dependencies")

# MagicMock for complex objects
def test_with_magic_mock() -> None:
    """MagicMock automatically handles magic methods."""
    mock_list = MagicMock()
    mock_list.__len__.return_value = 5
    mock_list.__getitem__.return_value = "item"
    
    assert len(mock_list) == 5
    assert mock_list[0] == "item"
    print("MagicMock handles __len__, __getitem__, etc.")

# Run demo tests
test_get_user_name_with_mock()
test_with_magic_mock()

print("\\nMocking best practices:")
print("  - Mock external dependencies, not internal code")
print("  - Verify mock was called correctly")
print("  - Use spec= to catch API mismatches")
print("  - Don't over-mock; prefer integration tests when possible")''',
                "explanation": "Mocking replaces real dependencies with test doubles. Mock objects track calls and can return predefined values. Use patch() to replace modules or objects temporarily during tests."
            }
        ]
    }
}


# =============================================================================
# DASH APPLICATION
# =============================================================================

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)

app.title = "Python Review Dashboard"


def create_sidebar() -> dbc.Nav:
    """Create the navigation sidebar with all topics."""
    nav_items = [
        dbc.NavLink(
            topic,
            href="#",
            id={"type": "topic-link", "topic": topic},
            className="nav-link-custom",
            active="exact"
        )
        for topic in PYTHON_EXAMPLES.keys()
    ]
    
    return dbc.Nav(
        nav_items,
        vertical=True,
        pills=True,
        className="flex-column"
    )


def create_example_card(example: dict[str, str], index: int) -> dbc.Card:
    """
    Create a card component for a single example.
    
    Args:
        example: Dictionary containing title, code, and explanation
        index: Example index for unique IDs
        
    Returns:
        A Dash Bootstrap Card component
    """
    return dbc.Card([
        dbc.CardHeader(
            html.H5(example["title"], className="mb-0"),
            className="bg-primary text-white"
        ),
        dbc.CardBody([
            html.H6("Code:", className="text-muted"),
            dcc.Markdown(
                f"```python\n{example['code']}\n```",
                className="code-block"
            ),
            html.Hr(),
            html.H6("Explanation:", className="text-muted"),
            html.P(example["explanation"], className="explanation-text")
        ])
    ], className="mb-4 shadow-sm")


def create_topic_content(topic: str) -> html.Div:
    """
    Create the content display for a selected topic.
    
    Args:
        topic: The topic name to display
        
    Returns:
        A Div containing all examples for the topic
    """
    if topic not in PYTHON_EXAMPLES:
        return html.Div("Select a topic from the sidebar")
    
    topic_data = PYTHON_EXAMPLES[topic]
    
    return html.Div([
        html.H2(topic, className="mb-3 text-primary"),
        html.P(topic_data["description"], className="lead mb-4"),
        html.Hr(),
        html.Div([
            create_example_card(example, i)
            for i, example in enumerate(topic_data["examples"])
        ])
    ])


# Layout
app.layout = dbc.Container([
    # Header
    html.Div([
        html.H1("Python Review Dashboard", className="mb-2"),
        html.P(
            "Interactive examples covering Python fundamentals",
            className="mb-0 opacity-75"
        )
    ], className="header"),
    
    # Main content
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.Div([
                html.H5("Topics", className="mb-3 text-muted"),
                create_sidebar()
            ], className="sidebar")
        ], width=3),
        
        # Content area
        dbc.Col([
            html.Div(
                id="content-area",
                className="main-content",
                children=create_topic_content("Variables & Data Types")
            )
        ], width=9)
    ])
], fluid=True, className="py-3")


@callback(
    Output("content-area", "children"),
    Input({"type": "topic-link", "topic": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def update_content(n_clicks: list[int | None]) -> html.Div:
    """
    Update the content area when a topic is clicked.
    
    Args:
        n_clicks: List of click counts for each topic link
        
    Returns:
        Updated content for the selected topic
    """
    if not ctx.triggered:
        return no_update
    
    # Get the clicked topic from the trigger
    triggered_id = ctx.triggered[0]["prop_id"]
    
    # Parse the topic from the pattern-matching ID
    import json
    try:
        # Extract JSON part from "{"type":"topic-link","topic":"..."}:n_clicks"
        json_str = triggered_id.rsplit(".", 1)[0]
        id_dict = json.loads(json_str)
        topic = id_dict["topic"]
        return create_topic_content(topic)
    except (json.JSONDecodeError, KeyError):
        return no_update


@callback(
    Output({"type": "topic-link", "topic": ALL}, "active"),
    Input({"type": "topic-link", "topic": ALL}, "n_clicks"),
    State({"type": "topic-link", "topic": ALL}, "id"),
    prevent_initial_call=True
)
def update_active_link(
    n_clicks: list[int | None],
    ids: list[dict]
) -> list[bool]:
    """
    Update the active state of topic links.
    
    Args:
        n_clicks: List of click counts
        ids: List of link IDs
        
    Returns:
        List of boolean active states
    """
    if not ctx.triggered:
        return [False] * len(ids)
    
    triggered_id = ctx.triggered[0]["prop_id"]
    
    import json
    try:
        json_str = triggered_id.rsplit(".", 1)[0]
        id_dict = json.loads(json_str)
        clicked_topic = id_dict["topic"]
        return [link_id["topic"] == clicked_topic for link_id in ids]
    except (json.JSONDecodeError, KeyError):
        return [False] * len(ids)


def main() -> None:
    """Run the Dash application."""
    print("Starting Python Review Dashboard...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True)


if __name__ == "__main__":
    main()
