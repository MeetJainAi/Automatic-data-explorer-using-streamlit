import seaborn as sns
import os
import streamlit as st
import pandas as pd
import os
import glob
import shutil
from PIL import Image
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def main():
    """ Automatic Dataset Explorer"""
    st.title("Automatic Dataset Explorer")
    st.subheader("Simple DataSet Explorer With StreamLit")

    html_bg = """ 
    <body style="background-color: yellowgreen;">
    </body>
    """

    html_temp = """
    <div style="background-color: tomato;">
    <p style="color: whitesmoke; font-size: 25px;">
        NO CODE JUST CLICKS TO EXPLORE ME!!
    </p>
</div>
    
    """
    st.markdown(html_bg, unsafe_allow_html=True)
    st.markdown(html_temp, unsafe_allow_html=True)

    def file_selector(folder_path='./datasets'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("Pick A File: ", filenames)
        return os.path.join(folder_path, selected_filename)

    filename = file_selector()
    st.info("You have selected {} :" .format(filename))

    # Read Data
    df = pd.read_csv(filename)
    # Show Dataset
    if st.checkbox("Show Dataset"):
        number = int(st.number_input(
            "Number of Rows to View:", 1, df.shape[0]))
        st.dataframe(df.head(number))
    # Show Columns
    if st.button('column Names'):
        st.write(df.columns)
    # Show shape
    if st.button("Show Shape"):
        shape = df.shape
        st.write("There are {} rows and {} columns" .format(
            shape[0], shape[1]))

        data_dim = st.radio("Show Dimension by", ("Rows", "Columns"))
        if data_dim == 'Rows':
            st.text("Number of  Rows")
            st.write(df.shape[0])
        elif data_dim == 'Columns':
            st.text("Number of Columns")
            st.write(df.shape[1])
    # Show columns by selection
    if st.checkbox("Select Columns To Show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect('Select', all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    # show Values
    if st.button("Value counts"):
        st.text('Value Counts By Target/Class')
        st.write(df.iloc[:, -1].value_counts())
    # Show datatypes
    if st.button("Data Types"):
        st.text('Data Types')
        st.write(df.dtypes)

    # Show Summary\
    if st.checkbox("Summary"):
        st.write(df.describe().T)

    #Plot and visualization

    st.subheader("Data Visualiztion")
    if st.checkbox("Correlation Plot [Matplotlib]"):
        plt.matshow(df.corr())
        st.pyplot()

    # Seaborn Plot
    if st.checkbox("Correlation Plot with Annotation[Seaborn]"):
        st.write(sns.heatmap(df.corr(), annot=True))
        st.pyplot()

    # Counts Plots
    if st.checkbox("Plot of Value Counts"):
        st.text("Value Counts By Target/Class")

        all_columns_names = df.columns.tolist()
        primary_col = st.selectbox(
            'Select Primary Column To Group By', all_columns_names)
        selected_column_names = st.multiselect(
            'Select Columns', all_columns_names)
        if st.button("Plot"):
            st.text("Generating Plot for: {} and {}".format(
                primary_col, selected_column_names))
            if selected_column_names:
                vc_plot = df.groupby(primary_col)[
                    selected_column_names].count()
            else:
                vc_plot = df.iloc[:, -1].value_counts()
            st.write(vc_plot.plot(kind='bar'))
            st.pyplot()

    # Pie Plot
    if st.checkbox("Pie Plot"):
        all_columns_names = df.columns.tolist()
        # st.info("Please Choose Target Column")
        # int_column =  st.selectbox('Select Int Columns For Pie Plot',all_columns_names)
        if st.button("Generate Pie Plot"):
            # cust_values = df[int_column].value_counts()
            # st.write(cust_values.plot.pie(autopct="%1.1f%%"))
            st.write(df.iloc[:, -1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

    # Barh Plot
    if st.checkbox("BarH Plot"):
        all_columns_names = df.columns.tolist()
        st.info("Please Choose the X and Y Column")
        x_column = st.selectbox(
            'Select X Columns For Barh Plot', all_columns_names)
        y_column = st.selectbox(
            'Select Y Columns For Barh Plot', all_columns_names)
        barh_plot = df.plot.barh(x=x_column, y=y_column, figsize=(10, 10))
        if st.button("Generate Barh Plot"):
            st.write(barh_plot)
            st.pyplot()

    # Customizable Plot
    st.subheader("Customizable Plot")
    all_columns = df.columns.to_list()
    plots = ['area', 'bar', 'line', 'hist', 'box', 'kde']
    type_of_plot = st.selectbox("Select type of plot:", plots)
    selected_column_names = st.multiselect(
        "select Columns to plot", all_columns)
    cust_target = df.iloc[:, -1].name
    if st.button("Generate Plot"):
        st.success("Generating A Customizable Plot of: {} for :: {}".format(
            type_of_plot, selected_column_names))
    # Plot By Streamlit
        if type_of_plot == 'area':
            cust_data = df[selected_column_names]
            st.area_chart(cust_data)
        elif type_of_plot == 'bar':
            cust_data = df[selected_column_names]
            st.bar_chart(cust_data)
        elif type_of_plot == 'line':
            cust_data = df[selected_column_names]
            st.line_chart(cust_data)
        elif type_of_plot == 'hist':
            custom_plot = df[selected_column_names].plot(
                kind=type_of_plot, bins=2)
            st.write(custom_plot)
            st.pyplot()
        elif type_of_plot == 'box':
            custom_plot = df[selected_column_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.pyplot()
        elif type_of_plot == 'kde':
            custom_plot = df[selected_column_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.pyplot()
        else:
            cust_plot = df[selected_column_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()

    st.subheader("Our Features and Target")

    if st.checkbox("Show Features"):
        all_features = df.iloc[:, 0:-1]
        st.text('Features Names:: {}'.format(all_features.columns[0:-1]))
        st.dataframe(all_features.head(10))

    if st.checkbox("Show Target"):
        all_target = df.iloc[:, -1]
        st.text('Target/Class Name:: {}'.format(all_target.name))
        st.dataframe(all_target.head(10))

    # Make Downloadable file as zip,since markdown strips to html
    st.markdown("""[google.com](iris.zip)""")

    st.markdown("""[google.com](./iris.zip)""")

    # def make_zip(data):
    # 	output_filename = '{}_archived'.format(data)
    # 	return shutil.make_archive(output_filename,"zip",os.path.join("downloadfiles"))

    def makezipfile(data):
        output_filename = '{}_zipped.zip'.format(data)
        with ZipFile(output_filename, "w") as z:
            z.write(data)
        return output_filename

    if st.button("Download File"):
        DOWNLOAD_TPL = f'[{filename}]({makezipfile(filename)})'
        # st.text(DOWNLOAD_TPL)
        st.text(DOWNLOAD_TPL)
        st.markdown(DOWNLOAD_TPL)


if __name__ == "__main__":
    main()
