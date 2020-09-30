# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:13:46 2020

@author: Ruben Andre Barreiro
"""

# Define a String called 'first_string',
# with the content "single line string"
first_string = "single line string"

# Define a String called 'second_string',
# with the content "this string\nhas two lines",
# where '\n' represents a line-break
second_string = '''this string
has two lines'''

# This is an echo of the String called 'first_string',
# so it doesn't print directly from a Python's Object,
# but you can run it directly in a Jupyter Notebook,
# IPython's Object, or similar
first_string

# This is an echo of the String called 'second_string',
# so it doesn't print directly from a Python's Object,
# but you can run it directly in a Jupyter Notebook,
# IPython's Object, or similar
second_string


# Note: But from the two previous examples,
#       you can print the contents from the Strings

# Print the String called 'first_string'
# with the content "single line string"
print(first_string)

# Print the String called 'second_string'
# with the content "this string\nhas two lines",
# where '\n' represents a line-break
print(second_string)

# List the methods belonging to an Object,
# the String called 'first_string', in this case
# This method is similar to the echo, previously explained,
# so, it doesn't print directly from a Python's Object,
# but you can run it directly in a Jupyter Notebook,
# IPython's Object, or similar
dir_firs_string = dir(first_string)

# But, once again, you can print
# the List of the methods belonging to an Object,
# the String called 'first_string'
print(dir_firs_string)

# The following instructions in Python
# are commands, so it doesn't run directly from
# the Python's Object, but you can run it directly in
# a Jupyter Notebook, IPython's Object, or similar

# Convert the String called 'first_string'
# with the content "single line string",
# to the Upper Case format
first_string.upper()

# Verify if the String called 'first_string',
# starts with 'single' String sub-part
first_string.startswith('single')

# Verify if the String called 'first_string',
# starts with 'abc' String sub-part
first_string.startswith('abc')

# Verify if the String 'abc',
# starts with 'a' String sub-part
'abc'.startswith('a')

# Convert the String 'ABC',
# to the Lower Case format
'ABC'.lower()

# Show the method to convert the String 'ABC',
# to the Lower Case format
'ABC'.lower

# But, once again, you can print the previous comands,
# assigning their contents to variables

# Convert the String called 'first_string'
# with the content "single line string",
# to the Upper Case format
first_string_upper_case = first_string.upper()

# Verify if the String called 'first_string',
# starts with 'single' String sub-part
first_string_starts_with_single = first_string.startswith('single')

# Verify if the String called 'first_string',
# starts with 'abc' String sub-part
first_string_starts_with_abc = first_string.startswith('abc')

# Verify if the String 'abc',
# starts with 'a' String sub-part
abc_string_starts_with_a = 'abc'.startswith('a')

# Convert the String 'ABC',
# to the Lower Case format
abc_upper_case_in_lower_case = 'ABC'.lower()

# Show the method to convert the String 'ABC',
# to the Lower Case format
abc_upper_case_in_lower_case_method = 'ABC'.lower

# Now, print the contents of
# the previously defined variables
print(first_string_upper_case)
print(first_string_starts_with_single)
print(first_string_starts_with_abc)
print(abc_string_starts_with_a)
print(abc_upper_case_in_lower_case)
print(abc_upper_case_in_lower_case_method)

# Work with the Strings' Slices,
# from their String Arrays

# Here, it will be used the String called 'first_string',
# as example, accessing its char positions,
# through the String Array

# Accessing the 1st Position of the String called 'first_string'
first_string[0]

# Accessing the 2nd Position of the String called 'first_string'
first_string[1]

# Accessing from the 3rd to 5th Position (Exclusive) of
# the String called 'first_string', i.e., [3,5[
first_string[2:4]

# Accessing from the last 4 Positions of
# the String called 'first_string',
# starting from the end
first_string[-4:]

# Accessing all the Positions of
# the String called 'first_string',
# starting from the 5th Position
first_string[4:]

# Accessing all the Positions of
# the String called 'first_string',
# until the 4th Position
first_string[:4]


# But, once again, you can print the previous comands,
# assigning their contents to variables

# Accessing the 1st Position of the String called 'first_string'
first_string_1st_position = first_string[0]

# Accessing the 2nd Position of the String called 'first_string'
first_string_2nd_position = first_string[1]

# Accessing from the 3rd to 5th Position
# (Exclusive, i.e., 4th Position) of
# the String called 'first_string', i.e., [3,5[
first_string_3rd_to_4th_position = first_string[2:4]

# Accessing from the last 4 Positions of
# the String called 'first_string',
# starting from the end
first_string_last_4_positions = first_string[-4:]

# Accessing all the Positions of
# the String called 'first_string',
# starting from the 5th Position
first_string_start_from_5th_position = first_string[4:]

# Accessing all the Positions of
# the String called 'first_string',
# until the 4th Position
first_string_until_4th_position = first_string[:4]


# Now, print the contents of
# the previously defined variables
print(first_string_1st_position)
print(first_string_2nd_position)
print(first_string_3rd_to_4th_position)
print(first_string_last_4_positions)
print(first_string_start_from_5th_position)
print(first_string_until_4th_position)


# Define different Objects and Structures
nothing = None # no value
minimum_wage = 505 # integer
pi_squared = 9.8696 # float
first_name = 'Ludwig' # string
years = [1999 , 2000 , 2001] # list
coordinates = (23.4 , 12.6 , 13.5) # tuple
animal_classes = {'fox':'mammal', # dictionary
                  'snake':'reptile',
                  'fly':'insect'}

# Define more different Objects and Structures,
# and operate them

# Create a List = [ 1 , 2 , 3 ]
a_list = [1 , 2 , 3] # list

# Print the List
print(a_list)

# Change the 2nd Position, from '2' to '4'
# List = [ 1 , 2 , 3 ] => List = [ 1 , 4 , 3 ]
a_list[1] = 4 

# Print the List modified
print(a_list)

# Create a String = 'abc'
a_string = 'abc' # string

# Print the String
print(a_string)

# Change the 2nd Position, from 'b' to 'x'
# String = 'abc' => String = 'axc'
# Note: It will raise an exception,
#       uncomment to try it
#a_string[1] = 'x'

# Print the String modified
# Note: It will raise an exception,
#       uncomment to try it
#print(a_string)

# Create a Tuple = ( 1 , 2 , 3 )
a_tuple = (1 , 2 , 3)

# Print the Tuple
print(a_tuple)

# Change the 2nd Position, from '2' to '4'
# Tuple = (1 , 2 , 3) => Tuple = (1 , 4 , 3)
# Note: It will raise an exception,
#       uncomment to try it
#a_tuple[1] = 4

# Print the Tuple modified
# Note: It will raise an exception,
#       uncomment to try it
#print(a_tuple)