import streamlit as st
import pandas as pd
import sys
sys.path.append('../')

from functions import *
from utils import *
import cv2

# functions
def label_decoder_int(line):
    '''Convert label from array to integer.
    
    '''
    if np.array_equal(line, [1,0,0,0,0]):
        return -1
    elif np.array_equal(line, [0,1,0,0,0]):
        return -0.5
    elif np.array_equal(line, [0,0,1,0,0]):
        return 0
    elif np.array_equal(line, [0,0,0,1,0]):
        return 0.5
    elif np.array_equal(line, [0,0,0,0,1]):
        return 1
    else:
        print('Error: Label not recognized.')
        return 

st.title("Axionaut Demo")

demos = [
    "",
    "Data balancing",
    "Data augmentation",
]

selected_demo = st.sidebar.selectbox("Select a Demo", demos)

if selected_demo == demos[1]:
    img = cv2.imread("test_image.jpg")
    st.write("Original image")
    st.image(img)

    list_of_functions = ["","Random Shadow","Night Effect", "Brightness"]
    aug_function = st.selectbox("test", list_of_functions)

    if aug_function == "Random Shadow":
        img_shadowed, _ = add_random_shadow(img, [1,0,0,0,0])
        st.image(img_shadowed)

    if aug_function == "Night Effect":
        img_night, _ = night_effect(img,[1,0,0,0,0])
        st.image(img_night)

    if aug_function == "Brightness":
        img_bright, _ = augment_brightness_camera_images(img, [1,0,0,0,0])
        st.image(img_bright)

if selected_demo == demos[2]:
    dataset_paths = [
        "PATH_INPUT_1","PATH_INPUT_2"
        ]

    # Load training data
    # Get training data from the Axionable track
    number_of_images = st.text_input("Number of images to extract", "")
    if number_of_images != "":
        nb_images = int(float(number_of_images))
        X, Y = get_images(dataset_paths,n_images=nb_images)
        Y = from_continue_to_discrete(Y)
        Y = Y[0]
        df = pd.DataFrame(Y)
        df['direction'] = df.apply(label_decoder_int, axis=1)
        counts = df.direction.value_counts()
        st.write("Car wheel data raw distribution")
        st.bar_chart(counts, width=100)

        st.markdown("Choose how many samples you would like to have per class")
        hard_turn_input = st.text_input("Hard turns", "")
        if hard_turn_input != "":
            hard_turn_number = int(float(hard_turn_input))
            hard_left = df[df.direction == -1].sample(n=hard_turn_number)
            hard_right = df[df.direction == 1].sample(n=hard_turn_number)

        soft_turn_input = st.text_input("Soft turns", "")
        if soft_turn_input != "":
            soft_turn_number = int(float(soft_turn_input))
            soft_left = df[df.direction == -0.5].sample(n=soft_turn_number)
            soft_right = df[df.direction == 0.5].sample(n=soft_turn_number)

        straight_input = st.text_input("Straight", "")
        if straight_input != "":
            straight_number = int(float(straight_input))
            straight = df[df.direction == 0].sample(n=straight_number)

            all_samples = hard_left.append(hard_right).append(soft_left).append(soft_right).append(straight)

            counts_balanced = all_samples.direction.value_counts()
            st.write("Car wheel data balanced distribution")
            st.bar_chart(counts_balanced, width=100)