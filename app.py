import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("train_ds.csv")

st.set_page_config(
    page_title="Titanic Dashboard",
    layout="wide"
)

st.title("🚢 Titanic Analytics Dashboard")

st.markdown(
    "Data Analysis and Visualization using Pandas, Seaborn, Matplotlib and Plotly"
)


df['Status'] = df['Survived'].map({
    0: 'Died',
    1: 'Survived'
})

col1, col2 = st.columns(2)

with col1:

    fig1, ax = plt.subplots()

    sns.countplot(
        x='Survived',
        data=df,
        color='brown',
        ax=ax
    )

    ax.set_title("Survival Distribution")

    st.pyplot(fig1)

with col2:

    fig2, ax = plt.subplots()

    sns.countplot(
        x='Sex',
        hue='Status',
        data=df,
        ax=ax,
        legend = ['Dead', 'Survived']
    )

    ax.set_title("Gender vs Survival")

    st.pyplot(fig2)


col1, col2 = st.columns(2)

with col1:

    survival_gender = pd.crosstab(
        df['Sex'],
        df['Status'],
        normalize='index'
    ) * 100

    fig3, ax = plt.subplots(figsize=(7,4))

    survival_gender.plot(
        kind='bar',
        ax=ax
    )

    ax.set_title("Gender Survival Percentage")

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.1f%%'
        )

    st.pyplot(fig3)

with col2:

    fig4, ax = plt.subplots()

    sns.histplot(
        df['Age'],
        bins=20,
        color='maroon',
        ax=ax
    )

    ax.set_title("Age Distribution")

    st.pyplot(fig4)


col1, col2 = st.columns(2)

with col1:

    fig5, ax = plt.subplots()

    sns.countplot(
        x='Pclass',
        hue='Status',
        data=df,
        ax=ax
    )

    ax.set_title("Class vs Survival")

    st.pyplot(fig5)

with col2:

    class_survival = pd.crosstab(
        df['Pclass'],
        df['Status'],
        normalize='index'
    ) * 100

    fig6, ax = plt.subplots()

    class_survival.plot(
        kind='bar',
        ax=ax
    )

    ax.set_title("Class Survival Percentage")

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.1f%%'
        )

    st.pyplot(fig6)


col1, col2 = st.columns(2)

with col1:

    fig7, ax = plt.subplots()

    sns.boxplot(
        x='Status',
        y='Fare',
        data=df,
        ax=ax
    )

    ax.set_title("Fare vs Survival")

    st.pyplot(fig7)


with col2:

    fig8 = px.histogram(
            df,
            x='Sex',
            color='Status',
            title='Gender Wise Survival'
    )

    st.plotly_chart(
        fig8,
        use_container_width=True
    )

fig9 = px.histogram(
    df,
    x='Pclass',
    color='Status',
    title='Class Wise Survival'
)

st.plotly_chart(
    fig9,
    use_container_width=True
)


st.header("Key Insights")

st.markdown("""
✔ Female passengers had higher survival rates.

✔ First-class passengers survived more often.

✔ Higher fares were generally associated with survival.

✔ Most passengers were between 20 and 40 years old.
""")