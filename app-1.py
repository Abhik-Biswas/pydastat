import numpy as np
import pandas as pd
import streamlit as st

input_file = st.file_uploader('Upload Dataset', type=['csv'])
if input_file is not None:
    df = pd.read_csv(input_file, encoding = 'ISO-8859-1')

    #df_10 = df.iloc[:10,:]

    st.dataframe(df.head(50))

    st.write('Numerical Columns Detected are: ')
    num_cols = []
    for i in list(df.columns):
        if df[i].dtype in [np.int32, np.int64, np.float32, np.float64]:
            num_cols.append(i)

    string = ""
    j = 1
    for i in num_cols:
        string+=f"({j})"+" "+i+" "
        j+=1
    st.write(f"{string}")
    st.sidebar.write('Choose numerical columns which you want to further process: ')

    selected = []
    for i in num_cols:
        selected.append(st.sidebar.checkbox(i, value=True))
    # st.write(selected)

    selected_cols = []
    for i in range(len(selected)):
        if selected[i]==True:
            selected_cols.append(num_cols[i])

    #st.write(selected_cols)


    st.markdown('### Check the distribution of Data:')

    col = st.selectbox('Select a column', selected_cols, key=1)

    bin_size=st.number_input('Enter the bin_size:')


    # st.write(col)

    import plotly.figure_factory as ff
    x = df[col]

    if len(list(x.dropna()))<df.shape[0]:
        y = np.array(x.dropna())
    else:
        y = np.array(x)

    labels = [col]


    fig=ff.create_distplot([y], labels, bin_size=bin_size)
    st.plotly_chart(fig)

    st.markdown('### Numerical Summary')
    col = st.selectbox('Select a column', selected_cols, key=2)

    st.write('Five Point Summary')

    x = df[col]

    if len(list(x.dropna()))<df.shape[0]:
        y = np.array(x.dropna())
    else:
        y = np.array(x)

    st.write(f"(1) Minimum: {np.min(y)} (2) Q1: {np.quantile(y, 0.25)} (3) Median: {np.quantile(y, 0.50)} (4) Q3: {np.quantile(y, 0.75)} (5) Maximum: {np.max(y)}")
    
    correl_type = st.selectbox('Select the Correlation Type', ['pearson', 'kendall', 'spearman'], key=3)
    st.markdown("### Correlation between Numerical Columns")

    corr_mat = df.loc[:,selected_cols].corr(method=correl_type).values 

    import plotly.express as px
    fig = px.imshow(corr_mat, labels=dict(x="Column Name", y="Column Name", color="Correlation"),
                    x=selected_cols, y=selected_cols, aspect='auto')
    fig.update_xaxes(side='top')
    st.plotly_chart(fig)

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    st.markdown('### Visualizing Relations amongst the Numerical Columns (Supporting Max of 4 Columns)')
    
    st.sidebar.write('Choose a subset of size less that or equal to 4 for mutivariate analysis')

    pair_cols = []

    k_b = 22
    for i in range(len(selected_cols)):
        pair_cols.append(st.sidebar.checkbox(selected_cols[i], value=False))
    
    #st.write(pair_cols)

    pair_cols1 = []
    for i in range(len(selected_cols)):
        if pair_cols[i]==True:
            pair_cols1.append(selected_cols[i])
    if len(pair_cols1)>=2:
        fig, ax = plt.subplots(len(pair_cols1), len(pair_cols1),
                           figsize=(len(pair_cols1)*4, len(pair_cols1)*4))
        for i in range(len(pair_cols1)):
            for j in range(len(pair_cols1)):
                if i>j:
                    ax[i,j].scatter(df[pair_cols1[i]], df[pair_cols1[j]])
                elif i==j:
                    sns.histplot(df[pair_cols1[i]], ax=ax[i,j], kde=True)
        st.pyplot(fig)
    
    

    else:
        pass 

    st.markdown('### Categorical Columns Analysis')

    categorical_cols = [i for i in df.columns if not i in selected_cols]

    #st.write(categorical_cols)

    x_ax = st.selectbox('Select Categorical Column for X-axis', categorical_cols, key=5)

    y_ax = st.selectbox('Select Numerical Column for Y-axis', selected_cols, key=6)

    c_cat = st.selectbox('Select Categorical Variable for color', ['None']+categorical_cols, key=7)

    type_chart = st.selectbox('Select Chart type', ['Boxplot', 'Violinplot'], key=8)

    if type_chart=='Boxplot':
        if c_cat !='None':
            fig = px.box(df.dropna(), x=x_ax, y=y_ax, color=c_cat)
            st.plotly_chart(fig)
        elif c_cat=='None':
            fig = px.box(df.dropna(), x=x_ax, y=y_ax)
            st.plotly_chart(fig)
    elif type_chart=='Violinplot':
        if c_cat !='None':
            fig = px.violin(df.dropna(), x=x_ax, y=y_ax, color=c_cat, box=True)
            st.plotly_chart(fig)
        elif c_cat=='None':
            fig = px.violin(df.dropna(), x=x_ax, y=y_ax, box=True)
            st.plotly_chart(fig)
    st.markdown('## Hypothesis Testing on Numerical Columns')

    test_type = st.selectbox('Select Hypothesis Test Type:', ['Z-Test', 'T-Test', 'Chi-Square Test', 'F-Test', '2-Sample Z-Test'])

    if test_type:
        col1 = st.selectbox('Select the first numerical column:', selected_cols)
        col2 = st.selectbox('Select the second numerical column (for 2-Sample Z-Test):', selected_cols)

    if test_type == 'Z-Test':
        from statsmodels.stats.weightstats import ztest
        st.write('Performing Z-Test:')
        test_statistic, p_value = ztest(df[col1].dropna(), alternative='two-sided')
        st.write(f'Test Statistic: {test_statistic}')
        st.write(f'P-Value: {p_value}')

    elif test_type == 'T-Test':
        from scipy.stats import ttest_ind
        st.write('Performing T-Test:')
        test_statistic, p_value = ttest_ind(df[col1].dropna(), df[col2].dropna())
        st.write(f'Test Statistic: {test_statistic}')
        st.write(f'P-Value: {p_value}')

    elif test_type == 'Chi-Square Test':
        from scipy.stats import chi2_contingency
        st.write('Performing Chi-Square Test:')
        contingency_table = pd.crosstab(df[col1].dropna(), df[col2].dropna())
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        st.write(f'Chi-Square Statistic: {chi2}')
        st.write(f'P-Value: {p}')

    elif test_type == 'F-Test':
        from scipy.stats import f_oneway
        st.write('Performing F-Test:')
        groups = [df[df[col1].notna()][col1], df[df[col2].notna()][col2]]  # Assuming col1 and col2 are categorical variables
        f_statistic, p_value = f_oneway(*groups)
        st.write(f'F-Statistic: {f_statistic}')
        st.write(f'P-Value: {p_value}')

    elif test_type == '2-Sample Z-Test':
        from statsmodels.stats.weightstats import ztest
        st.write('Performing 2-Sample Z-Test:')
        test_statistic, p_value = ztest(df[col1].dropna(), df[col2].dropna(), alternative='two-sided')
        st.write(f'Test Statistic: {test_statistic}')
        st.write(f'P-Value: {p_value}')
else:
    pass


