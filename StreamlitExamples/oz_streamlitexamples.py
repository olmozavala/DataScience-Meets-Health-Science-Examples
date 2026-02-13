import streamlit as st
import pandas as pd
import numpy as np
import time

# --- Page Configuration ---
# This must be the first Streamlit command in your script.
# It sets the title, icon, and layout of the page in the browser tab.
st.set_page_config(
    page_title="Streamlit Tutorial",
    page_icon="🚀",
    layout="wide",
)

# --- Sidebar Navigation ---
# We use a selectbox in the sidebar to create a simple multi-page feel.
# The sidebar is always accessible to the user.
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to",
    ["1. Welcome & Basics", "2. Working with Data", "3. Input Widgets", "4. Layout & Containers", "5. Charts & Media", "6. Session State & Caching"]
)

# --- Module 1: Welcome & Basics ---
if page == "1. Welcome & Basics":
    st.title("Welcome to the Streamlit Tutorial! 👋")
    
    st.markdown("""
    Streamlit is an open-source Python library that makes it easy to create and share custom web apps for machine learning and data science.
    
    ### How it works:
    1. **Python first:** Write apps in plain Python.
    2. **Instant updates:** Updates appear as soon as you save the script.
    3. **Interactive:** Widgets are easy to add and use.
    """)
    
    st.header("1. Core Text Elements")
    st.write("`st.write()` is the 'Swiss Army knife' of Streamlit. It can display text, data, plots, and more.")
    
    st.subheader("Subheaders and Text")
    st.text("This is simple preformatted text using st.text().")
    st.caption("This is a caption, often used for small notes or footnotes.")
    
    st.header("2. Markdown Support")
    st.markdown("""
    Streamlit supports **Markdown**, allowing you to format text easily:
    - *Italics*
    - **Bold**
    - [Links](https://streamlit.io)
    - Code blocks: `print('Hello')`
    
    You can even use LaTeX for mathematics:
    $$
    E = mc^2
    $$
    """)

# --- Module 2: Working with Data ---
elif page == "2. Working with Data":
    st.title("📊 Working with Data")
    
    st.header("1. Dataframes")
    # Generating sample data
    df = pd.DataFrame(
        np.random.randn(10, 5),
        columns=('col %d' % i for i in range(5))
    )
    
    st.write("You can use `st.dataframe()` to display an interactive table:")
    # st.dataframe allows sorting, resizing, and searching
    st.dataframe(df.style.highlight_max(axis=0))
    
    st.write("`st.table()` displays a static table (no sorting or interaction):")
    st.table(df.head())
    
    st.header("2. Metrics")
    # Metrics are great for displaying KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", "70 °F", "1.2 °F")
    col2.metric("Wind Speed", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")

    st.header("3. JSON Data")
    st.json({
        'foo': 'bar',
        'baz': 'boz',
        'stuff': [
            'stuff 1',
            'stuff 2',
            'stuff 3',
        ],
    })

# --- Module 3: Input Widgets ---
elif page == "3. Input Widgets":
    st.title("🔘 Input Widgets")
    st.write("Widgets allow users to interact with your app. When a widget is interacted with, the entire script reruns from top to bottom.")

    st.header("1. Buttons and Clicks")
    if st.button('Say hello'):
        st.write('Why hello there!')
    else:
        st.write('Goodbye')

    st.header("2. Selection Widgets")
    # Checkbox
    if st.checkbox('Show/Hide'):
        st.write('You checked the box!')

    # Radio buttons
    genre = st.radio(
        "What's your favorite movie genre?",
        ('Comedy', 'Drama', 'Documentary')
    )
    st.write(f"You selected: {genre}")

    # Selectbox
    option = st.selectbox(
        'How would you like to be contacted?',
        ('Email', 'Home phone', 'Mobile phone')
    )
    st.write('You selected:', option)

    # Multi-select
    options = st.multiselect(
        'What are your favorite colors?',
        ['Green', 'Yellow', 'Red', 'Blue'],
        ['Yellow', 'Red'] # Default values
    )
    st.write('You selected:', options)

    st.header("3. Numeric and Text Inputs")
    # Slider
    age = st.slider('How old are you?', 0, 130, 25)
    st.write("I'm ", age, 'years old')

    # Slider for range
    values = st.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0)
    )
    st.write('Values:', values)

    # Text Input
    title = st.text_input('Movie title', 'Life of Brian')
    st.write('The current movie title is', title)

    # Number Input
    number = st.number_input('Insert a number', value=0, step=1)
    st.write('The current number is ', number)

# --- Module 4: Layout & Containers ---
elif page == "4. Layout & Containers":
    st.title("🏗️ Layout & Containers")
    st.write("Control how your app is structured visually.")

    st.header("1. Columns")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Column 1")
        st.image("https://static.streamlit.io/examples/cat.jpg")
    with col2:
        st.header("Column 2")
        st.image("https://static.streamlit.io/examples/dog.jpg")

    st.header("2. Sidebar")
    st.write("Look over there! ⬅️ We've already used `st.sidebar` for navigation.")
    st.sidebar.markdown("---")
    st.sidebar.info("This is an info box in the sidebar!")

    st.header("3. Tabs")
    tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])
    with tab1:
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
    with tab2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
    with tab3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

    st.header("4. Expander")
    with st.expander("See explanation"):
        st.write("""
            The chart above shows some numbers I picked out of my head.
            Just like this text, it's hidden until you click.
        """)
        st.image("https://static.streamlit.io/examples/dice.jpg")

    st.header("5. Containers")
    with st.container():
        st.write("This is inside a container")
        # You can use containers to group elements or insert them out of order
        st.write("Containers don't have a visible border by default.")

# --- Module 5: Charts & Media ---
elif page == "5. Charts & Media":
    st.title("📈 Charts & Media")
    st.write("Streamlit has built-in support for several charting libraries and media types.")

    st.header("1. Built-in Charts")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c']
    )

    st.subheader("Line Chart")
    st.line_chart(chart_data)

    st.subheader("Area Chart")
    st.area_chart(chart_data)

    st.subheader("Bar Chart")
    st.bar_chart(chart_data)

    st.header("2. Map")
    # Generating coordinates around San Francisco
    map_data = pd.DataFrame(
        np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon']
    )
    st.map(map_data)

    st.header("3. Media")
    st.subheader("Audio")
    # Example audio (using a sample URL or local file)
    st.audio("https://www.w3schools.com/html/horse.ogg")

    st.subheader("Video")
    st.video("https://www.youtube.com/watch?v=R2nr1vZ8khc")

# --- Module 6: Session State & Caching ---
elif page == "6. Session State & Caching":
    st.title("🧠 Advanced Features")
    
    st.header("1. Session State")
    st.write("""
    `st.session_state` is a way to share variables between reruns, for each user session.
    It's like a dictionary that persists while the user interacts with the app.
    """)

    # Initialize session state if it doesn't exist
    if 'counter' not in st.session_state:
        st.session_state.counter = 0

    def increment_counter():
        st.session_state.counter += 1

    st.write(f"Counter value: **{st.session_state.counter}**")
    st.button('Increment', on_click=increment_counter)
    
    if st.button('Reset'):
        st.session_state.counter = 0
        st.rerun()

    st.header("2. Caching")
    st.write("""
    `@st.cache_data` allows you to cache the output of a function. 
    This is essential for expensive operations like loading large datasets or complex calculations.
    """)

    @st.cache_data
    def expensive_computation(a, b):
        st.write(f"Running expensive computation for {a} and {b}...")
        time.sleep(2) # Simulate a long-running process
        return a + b

    a = st.number_input("Input A", value=10)
    b = st.number_input("Input B", value=20)
    
    result = expensive_computation(a, b)
    st.write(f"Result: {result}")
    st.info("Try changing the inputs. Notice how it takes 2 seconds the first time, but is instant the second time!")

# Default case (should not be hit)
else:
    st.error("Invalid page selection.")
