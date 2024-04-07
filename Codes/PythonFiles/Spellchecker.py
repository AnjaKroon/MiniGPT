#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# @title Standaard titeltekst
get_ipython().system('pip install nltk')
get_ipython().system('pip install pyspellchecker ')
#!pip install language_tool_python


# In[ ]:


from spellchecker import SpellChecker


# In[ ]:


# Read the contents of the file
with open('input.txt', 'r') as file:
    text = file.read()


# In[ ]:


def calculate_misspelled_percentage(text_to_check_path):
    spell = SpellChecker()

    with open(text_to_check_path, 'r') as file:
        text = file.read()
    
    words = text.split()
    misspelled = spell.unknown(words)

    total_words = len(words)
    misspelled_count = len(misspelled)
    
    #for word in misspelled:
    #    print(f"Misspelled: {word} -> Suggestion: {spell.correction(word)}")
    
    if total_words > 0:
        correct_count = total_words - misspelled_count
        misspelled_percentage = (misspelled_count / total_words) * 100
        correct = 100 - misspelled_percentage
        print(f"Percentage of Correctly Spelled Words: {correct:.2f}%")
    else:
        print("The file is empty or no words to check.")
    return correct_count

baseline_correct = calculate_misspelled_percentage('input.txt')
generated_correct = calculate_misspelled_percentage('Generated.txt')        # CHANGE TO GENERATED TXT FILE

# "manage" there being old english by normalizing by the baseline correctness
# maybe also add words which are one removed from an actual english word like levinstein distance?
# will see if that is per se needed
# don't forget to normalize with the baseline text


# In[1]:


calculate_misspelled_percentage('Generated.txt')

