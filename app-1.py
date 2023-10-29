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
        col2 = st.selectbox('Select the second numerical column:', set(selected_cols)-{col1})
        # Option to enter custom values for the test
        custom_test_values = st.checkbox('Enter Custom Test Values', value=False)
    
        if custom_test_values:
            st.write('Enter Custom Test Values:')
            alpha = st.number_input('Significance Level (alpha):', min_value=0.001, max_value=0.5, step=0.001, value=0.05)
            sample_mean = st.number_input('Sample Mean:', value=0.0)
            sample_stddev = st.number_input('Sample Standard Deviation:', value=1.0)
    
        if test_type == 'Z-Test':
            from statsmodels.stats.weightstats import ztest
            from scipy.stats import norm
            st.write('Performing Z-Test:')
            if custom_test_values:
                test_statistic, p_value = ztest(df[col1].dropna(), value=sample_mean, alternative='two-sided', ddof=1)
            else:
                test_statistic, p_value = ztest(df[col1].dropna(), alternative='two-sided')
            plt.figure(figsize=(10, 6))
            x = np.linspace(-3, 3, 1000)
            plt.plot(x, norm.pdf(x, 0, 1), label='Standard Normal Distribution')
            plt.fill_between(x, 0, norm.pdf(x, 0, 1), where=(x < -test_statistic), color='red', alpha=0.3, label='Critical Region')
            plt.fill_between(x, 0, norm.pdf(x, 0, 1), where=(x > test_statistic), color='red', alpha=0.3)
            plt.axvline(test_statistic, color='blue', linestyle='--', label=f'Test Statistic = {test_statistic:.2f}')
            plt.legend()
            plt.title('Z-Test for Mean')
            plt.xlabel('Z-Score')
            plt.ylabel('Probability Density')
            st.pyplot(plt)
        
            st.write(f'Test Statistic: {test_statistic:.2f}')
            st.write(f'P-Value: {p_value:.4f}')
            if p_value < alpha:
                st.write(f'Reject the null hypothesis at alpha = {alpha}')
            else:
                st.write(f'Fail to reject the null hypothesis at alpha = {alpha}')
            

        elif test_type == 'T-Test':
            from scipy.stats import ttest_1samp, t
            st.write('Performing T-Test:')
            if custom_test_values:
                test_statistic, p_value = ttest_1samp(df[col1].dropna(), sample_mean)
            else:
                test_statistic, p_value = ttest_1samp(df[col1].dropna(), 0)

            df_degrees = len(df[col1].dropna()) - 1
            t_critical_right = t.ppf(1 - alpha / 2, df=df_degrees)
            t_critical_left = -t_critical_right
    
            # Create a plot to visualize the T-Test
            x = np.linspace(-3, 3, 1000)
            t_dist = t.pdf(x, df=df_degrees)
    
            plt.figure(figsize=(10, 6))
            plt.plot(x, t_dist, label=f'T-Distribution (df={df_degrees})')
            plt.fill_between(x, 0, t_dist, where=(x < t_critical_left), color='red', alpha=0.3, label='Critical Region')
            plt.fill_between(x, 0, t_dist, where=(x > t_critical_right), color='red', alpha=0.3)
            plt.axvline(test_statistic, color='blue', linestyle='--', label=f'Test Statistic = {test_statistic:.2f}')
            plt.legend()
            plt.title('T-Test for Mean')
            plt.xlabel('T-Score')
            plt.ylabel('Probability Density')
            st.pyplot(plt)
    
            st.write(f'Test Statistic: {test_statistic:.2f}')
            st.write(f'P-Value: {p_value:.4f}')
    
            if p_value < alpha:
                st.write(f'Reject the null hypothesis at alpha = {alpha}')
            else:
                st.write(f'Fail to reject the null hypothesis at alpha = {alpha}')
                
        elif test_type == 'Chi-Square Test':
            from scipy.stats import chi2_contingency, chi2
            st.write('Performing Chi-Square Test:')
            contingency_table = pd.crosstab(df[col1].dropna(), df[col2].dropna())
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            x = np.linspace(0, chi2 * 2, 1000)
            chi2_dist = chi2.pdf(x, dof)
    
            critical_value = chi2.ppf(1 - alpha, df=dof)
    
            plt.figure(figsize=(10, 6))
            plt.plot(x, chi2_dist, label=f'Chi-Square Distribution (df={dof})')
            plt.fill_between(x, 0, chi2_dist, where=(x > critical_value), color='red', alpha=0.3, label='Critical Region')
            plt.axvline(chi2, color='blue', linestyle='--', label=f'Chi-Square Statistic = {chi2:.2f}')
            plt.legend()
            plt.title('Chi-Square Test')
            plt.xlabel('Chi-Square Statistic')
            plt.ylabel('Probability Density')
            st.pyplot(plt)
    
            st.write(f'Chi-Square Statistic: {chi2:.2f}')
            st.write(f'P-Value: {p:.4f}')
    
            if p < alpha:
                st.write(f'Reject the null hypothesis at alpha = {alpha}')
            else:
                st.write(f'Fail to reject the null hypothesis at alpha = {alpha}')

        elif test_type == 'F-Test':
            from scipy.stats import f_oneway, f

            st.write('Performing F-Test for Two Variances:')
            var1 = df[col1].dropna().var()
            var2 = df[col2].dropna().var()

            # Calculate F-statistic
            f_statistic = var1 / var2 if var1 > var2 else var2 / var1

            dfn = len(df[col1].dropna()) - 1
            dfd = len(df[col2].dropna()) - 1
            f_critical = f.ppf(1 - alpha / 2, dfn=dfn, dfd=dfd)
    
            # Create a plot to visualize the F-Test for Two Variances
            x = np.linspace(0, f_critical * 2, 1000)
            f_dist = f.pdf(x, dfn=dfn, dfd=dfd)
    
            plt.figure(figsize=(10, 6))
            plt.plot(x, f_dist, label=f'F-Distribution (dfn={dfn}, dfd={dfd})')
            plt.fill_between(x, 0, f_dist, where=(x > f_critical), color='red', alpha=0.3, label='Critical Region')
            plt.axvline(f_statistic, color='blue', linestyle='--', label=f'F-Statistic = {f_statistic:.2f}')
            plt.legend()
            plt.title('F-Test for Two Variances')
            plt.xlabel('F-Statistic')
            plt.ylabel('Probability Density')
            st.pyplot(plt)
    
            st.write(f'F-Statistic: {f_statistic:.2f}')
            st.write(f'Critical Value at alpha/2: {f_critical:.4f}')
    
            if f_statistic > f_critical:
                st.write(f'Reject the null hypothesis at alpha = {alpha}')
            else:
                st.write(f'Fail to reject the null hypothesis at alpha = {alpha}')

        elif test_type == '2-Sample Z-Test':
            from statsmodels.stats.weightstats import ztest
            st.write('Performing 2-Sample Z-Test:')
            if custom_test_values:
                test_statistic, p_value = ztest(df[col1].dropna(), df[col2].dropna(), value=0.0, alternative='two-sided', ddof=1)
            else:
                test_statistic, p_value = ztest(df[col1].dropna(), df[col2].dropna(), alternative='two-sided')

            sample1 = df[col1].dropna()
            sample2 = df[col2].dropna()
    
            # Calculate the test statistic and p-value
            test_statistic, p_value = ztest(sample1, sample2, alternative='two-sided')
    
            # Calculate critical values for two-tailed test
            z_critical_left = norm.ppf(alpha / 2)
            z_critical_right = norm.ppf(1 - alpha / 2)
    
            # Create a plot to visualize the 2-Sample Z-Test
            x = np.linspace(-3, 3, 1000)
            z_dist = norm.pdf(x, 0, 1)
    
            plt.figure(figsize=(10, 6))
            plt.plot(x, z_dist, label='Standard Normal Distribution')
            plt.fill_between(x, 0, z_dist, where=(x < z_critical_left), color='red', alpha=0.3, label='Critical Region')
            plt.fill_between(x, 0, z_dist, where=(x > z_critical_right), color='red', alpha=0.3)
            plt.axvline(test_statistic, color='blue', linestyle='--', label=f'Test Statistic = {test_statistic:.2f}')
            plt.legend()
            plt.title('2-Sample Z-Test for Means')
            plt.xlabel('Z-Score')
            plt.ylabel('Probability Density')
            st.pyplot(plt)
    
            st.write(f'Test Statistic: {test_statistic:.2f}')
            st.write(f'P-Value: {p_value:.4f}')
    
            if p_value < alpha:
                st.write(f'Reject the null hypothesis at alpha = {alpha}')
            else:
                st.write(f'Fail to reject the null hypothesis at alpha = {alpha}')
    else:
        pass
else:
    pass


