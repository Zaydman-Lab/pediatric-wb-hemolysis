'''

figures.py

Created: Mark Zaydman 10/18/2022
Purpose: Generate figures for pediatric WB K hemolysis paper

'''

#%% Imports
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon,linregress

#%% read in data
def import_data(dpath:str)->pd.DataFrame:
    """Imports csv data from files and returns dataframe"""
    files=glob.glob(os.path.join(dpath,'*.csv'))
    df=pd.concat([pd.read_csv(file,escapechar='\\',index_col=0) for file in files])
    return(df)

def process_data(df,age_bins=[0,31/365,6*31/365,1,2],age_bins_names=['< 1 mo','1 mo - 6 mo','6 mo - 1 yr','1 yr - 2 yr']):
    """Returns processed dataframe"""
    df=df.drop_duplicates(subset=['ORDER_ID','TASK_ASSAY']).reset_index(drop=True)
    df['age']=(pd.to_datetime(df['DRAWN_DT_TM'])-pd.to_datetime(df['BIRTH_DT_TM']))/np.timedelta64(1,'Y')
    df['age_bin']=pd.cut(df['age'],age_bins,labels=age_bins_names)
    df['DRAWN_DT_TM']=pd.to_datetime(df['DRAWN_DT_TM'])
    df['drawn_yyyymm']=df['DRAWN_DT_TM'].dt.to_period('M')
    return(df)

def add_groupers(df):
    df['LOCATION_GROUPER']=np.NaN
    filt_outpatient=df.ENCOUNTER_TYPE_CLASS=='Outpatient'
    filt_inpatient=df.NURSE_UNIT.str.startswith(('9','10','11','12'),na=False)
    filt_nicu=df.NURSE_UNIT.str.startswith('5',na=False)
    filt_picu=df.NURSE_UNIT.isin(['8200','8100'])
    filt_cicu=df.NURSE_UNIT.str.startswith('7',na=False)
    filt_emerg=df.NURSE_UNIT=='EMERG'
    filt_nursery=df.NURSE_UNIT.isin(['NAC','NURSERY','6800N','SCC RON','SLC SUR','58LDN']) # NEED TO UPDATE
    df.loc[filt_outpatient,'LOCATION_GROUPER']='OUTPATIENT'
    df.loc[filt_nicu|filt_picu|filt_cicu,'LOCATION_GROUPER']='ICU'
    df.loc[filt_emerg,'LOCATION_GROUPER']='EMERGENCY'
    df.loc[filt_nursery,'LOCATION_GROUPER']='NURSERY'
    df.loc[filt_inpatient,'LOCATION_GROUPER']='INPATIENT'
    df.loc[df.NURSE_UNIT=='9SIC','LOCATION_GROUPER']='INPATIENT'
    return(df)


def plot_volumes(df_plasma:pd.DataFrame,df_wb:pd.DataFrame,ax=None):
    """Plots volumes of plasma and wb k"""
    if not ax:
        fig,ax=plt.subplots()
        fig.set_size_inches(8,4)
    pt=pd.concat([df_plasma.loc[df_plasma['TASK_ASSAY'].str.contains('Potassium')].groupby('drawn_yyyymm').agg(plasma=pd.NamedAgg('ACCESSION',pd.Series.nunique)),
        df_wb.loc[df_wb['TASK_ASSAY'].str.contains('Potassium')].groupby('drawn_yyyymm').agg(wb=pd.NamedAgg('ACCESSION',pd.Series.nunique))]
        ,axis=1).reset_index().melt(id_vars='drawn_yyyymm',var_name='Specimen',value_name='# accessions').sort_values(by='drawn_yyyymm',ascending=True)
    
    g=sns.barplot(
        x=pt['drawn_yyyymm'],
        y=pt['# accessions'],
        hue=pt['Specimen'],
        ax=ax,
        palette=sns.color_palette("colorblind")
    )
    plt.xticks(rotation=90,fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('')
    

def plot_locgroupvolumes(df_plasma,df_wb,ax=None):
    """Plots volumes from grouped collection locations"""
    if not ax:
        fig,ax=plt.subplots()
        fig.set_size_inches(4,3)
    pt=pd.concat([df_plasma.loc[df_plasma['TASK_ASSAY'].str.contains('Potassium')].groupby('LOCATION_GROUPER').agg(plasma=pd.NamedAgg('ACCESSION',pd.Series.nunique)),
        df_wb.loc[df_wb['TASK_ASSAY'].str.contains('Potassium')].groupby('LOCATION_GROUPER').agg(wb=pd.NamedAgg('ACCESSION',pd.Series.nunique))]
        ,axis=1).reset_index().melt(id_vars='LOCATION_GROUPER',var_name='Specimen',value_name='# accessions').sort_values(by='LOCATION_GROUPER',ascending=True)    
    sns.set_style("white")    
    g=sns.barplot(
        x=pt['LOCATION_GROUPER'],
        y=pt['# accessions'],
        hue=pt['Specimen'],
        ax=ax,
        palette='gray'
    )
    plt.xticks(rotation=90,fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('# specimen')
    ax.set_yscale('log')
    ax.set_yticks([1,10,100,1000,10000,100000])
    ax.set_yticklabels(['1','10','100','1,000','10,000','100,000'])
    plt.legend(loc='upper right')
    return(pt.pivot(index='LOCATION_GROUPER',columns=['Specimen'],values='# accessions'))

def plot_agegroupvolumes(df_plasma,df_wb,ax=None):
    """Plots volumes from grouped collection locations"""
    if not ax:
        fig,ax=plt.subplots()
        fig.set_size_inches(4,3)
    pt=pd.concat([df_plasma.loc[df_plasma['TASK_ASSAY'].str.contains('Potassium')].groupby('age_bin').agg(plasma=pd.NamedAgg('ACCESSION',pd.Series.nunique)),
        df_wb.loc[df_wb['TASK_ASSAY'].str.contains('Potassium')].groupby('age_bin').agg(wb=pd.NamedAgg('ACCESSION',pd.Series.nunique))]
        ,axis=1).reset_index().melt(id_vars='age_bin',var_name='Specimen',value_name='# accessions').sort_values(by='age_bin',ascending=True)    
    sns.set_style("white")    
    g=sns.barplot(
        x=pt['age_bin'],
        y=pt['# accessions'],
        hue=pt['Specimen'],
        ax=ax,
        palette='gray'
    )
    plt.xticks(rotation=90,fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('# specimen')
    return(pt.pivot(index='age_bin',columns=['Specimen'],values='# accessions'))


def plot_bycol(df,dta,ax,feature_col='MEDICAL_SERVICE',n_top=10,logscalex=False):
    set_top=set(df[feature_col].value_counts()[:n_top].index.values)
    filt_top=df[feature_col].isin(set_top)
    filt_dta=df['TASK_ASSAY']==dta

    g=sns.ecdfplot(data=df.loc[filt_top&filt_dta],
        x='RESULT_VALUE_NUMERIC',
        ax=ax,
        hue=feature_col,
        palette='gray',
        )    
    ax.legend([x.get_text() for x in g.get_legend().get_texts()],title=feature_col.lower().capitalize().split('_')[0])
    plt.setp(g.get_legend().get_texts(), fontsize='8') 
    plt.setp(g.get_legend().get_title(), fontsize='8') 
    ax.set_xlabel(dta)
    if logscalex:
        ax.set_xscale('log')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)    

def pct_50(X):
    return(np.nanpercentile(X,50))

def pct_90(X):
    return(np.nanpercentile(X,90))

def pct_95(X):
    return(np.nanpercentile(X,95))    

def above_5(X):
    return('%.1f%%'%(100*(sum(X>5))/len(X)))

def above_10(X):
    return('%.1f%%'%(100*(sum(X>10))/len(X)))

def above_50(X):
    return('%.1f%%'%(100*(sum(X>50))/len(X)))
    
def above_100(X):
    return('%.1f%%'%(100*(sum(X>100))/len(X)))

def percent_hyperK(X):
    if len(X)>0:
        return('%.1f%%'%(100*sum(X>5)/len(X)))
    else:
        return(np.NaN)
    
def percent_hypoK(X):
    if len(X)>0:
        return('%.1f%%'%(100*sum(X<3.3)/len(X)))
    else:
        return(np.NaN)    

def plot_luohcdf(df_plasma,ax=False,n_top=8)->pd.DataFrame:
    """Plots ecdfs of luoh values broken out by age and location type"""
    if not ax:
        fig,ax=plt.subplots(1,2,sharex=False)
        fig.set_size_inches(8,4)
    plot_bycol(df_plasma,'LUO-H',ax[0],'age_bin',n_top,logscalex=True)
    plot_bycol(df_plasma,'LUO-H',ax[1],'LOCATION_GROUPER',n_top,logscalex=True)
    ax[0].set_xlabel('Hemolysis index')
    ax[1].set_xlabel('Hemolysis index')
    res=df_plasma[df_plasma.TASK_ASSAY=='LUO-H'].pivot_table(
        index='age_bin',
        columns='LOCATION_GROUPER',
        values='RESULT_VALUE_NUMERIC',
        aggfunc=[pct_50,pct_90,pct_95,'count']
    ).fillna(0)
    res.columns=res.columns.swaplevel(0,1)
    res.sort_index(axis=1,level=0)
    return(res)

def plot_kcdf(df,dta,ax=False,n_top=8)->pd.DataFrame:
    """Plots ecdfs of plasma K broken out by age or loc type, returns results"""
    if not ax:
        fig,ax=plt.subplots(1,2,sharex=False)
        fig.set_size_inches(8,4)
    plot_bycol(df,dta,ax[0],'age_bin',n_top)
    plot_bycol(df,dta,ax[1],'LOCATION_GROUPER',n_top)
    ax[0].set_xlim([0,10])
    ax[1].set_xlim([0,10])
    res=df[df.TASK_ASSAY==dta].pivot_table(
        index='age_bin',
        columns='LOCATION_GROUPER',
        values='RESULT_VALUE_NUMERIC',
        aggfunc=[pct_50,pct_90,pct_95,'count']
    ).fillna(0)
    res.columns=res.columns.swaplevel(0,1)
    res.sort_index(axis=1,level=0)
    return(res)


def plot_kvluoh(df_plasma,ax=False):
    """Plots plasma K versus luoh"""
    joined=df_plasma.loc[df_plasma['TASK_ASSAY']=='LUO-H',['ACCESSION','RESULT_VALUE_NUMERIC']].merge(
    df_plasma.loc[df_plasma['TASK_ASSAY']=='Potassium Plas',['ACCESSION','RESULT_VALUE_NUMERIC']],on='ACCESSION',).rename(columns={'RESULT_VALUE_NUMERIC_x':'LUO-H','RESULT_VALUE_NUMERIC_y':'Potassium Plas'})

    joined['True hyperkalemia']=joined.apply(lambda x:(x['LUO-H']<50)&(x['Potassium Plas']>5),axis=1)
    joined['LUO-H bin']=pd.cut(joined['LUO-H'],np.append(np.arange(0,110,10),1000))

    if not ax:
        fig,ax=plt.subplots()
        
        
    slope, intercept, r_value, p_value, std_err = linregress(x=joined.dropna()['LUO-H'], y=joined.dropna()['Potassium Plas'])
    ax.annotate(f'$y={slope:.3f}x{intercept:+.2f}$\n$r^2 = {r_value ** 2:.2f}$',
                xy=(.05, .95), xycoords=ax.transAxes, fontsize=8,
                color='black', backgroundcolor='#FFFFFF00', ha='left', va='top')
            
    sns.regplot(data=joined.dropna(),
        x='LUO-H',
        y='Potassium Plas',
        ax=ax,
        scatter_kws={'alpha':0.1,
                     's':5,
                     'color':'gray',
                     'linewidth':0},
        line_kws={'color':'black'}
        )

    ax.set_xlim(0,100)
    ax.set_xlabel('Hemolysis index')
    ax.set_ylabel('Plasma K (mmol/L)')

    ax.spines.top.set_visible(False)    
    ax.spines.right.set_visible(False)    
    ax.hlines(3.3,1,1000,color='k',linestyle=':')
    ax.hlines(4.9,1,1000,color='k',linestyle=':')
    ax.vlines(50,0.0,10,color='k',linestyle=':')
    res=joined.pivot_table(
        index='LUO-H bin',
        values='Potassium Plas',
        aggfunc=['count',np.median,percent_hyperK,percent_hypoK]
    )
    return(res)

def plot_dtwbplas(df_plasma,df_wb,threshold=7*24,ax=False):
    """Returns dataframe with paired plasma and WB result based on nearest draw times"""
    cols=['PERSON_ID','DRAWN_DT_TM','COMPLETED_DT_TM','RESULT_VALUE_NUMERIC','ACCESSION','YRS_OLD_AT_COLLECTION']


    joined=df_plasma.loc[df_plasma['TASK_ASSAY']=='LUO-H',['ACCESSION','RESULT_VALUE_NUMERIC']].merge(
        df_plasma.loc[df_plasma['TASK_ASSAY']=='Potassium Plas',['ACCESSION','RESULT_VALUE_NUMERIC','PERSON_ID','DRAWN_DT_TM']],on='ACCESSION').rename(columns={'RESULT_VALUE_NUMERIC_x':'LUO-H','RESULT_VALUE_NUMERIC_y':'Potassium Plas','ACCESSION':'ACCESSION_PLASMA'})

    res=pd.merge_asof(
        df_wb.loc[df_wb['TASK_ASSAY']=='Potassium WB',cols].sort_values(by='DRAWN_DT_TM').rename(columns={'DRAWN_DT_TM':'DRAWN_DTTM_WB','COMPLETED_DT_TM':'COMPLETED_DT_TM_WB'}),
        joined[['PERSON_ID','Potassium Plas','LUO-H','DRAWN_DT_TM','ACCESSION_PLASMA']].sort_values(by='DRAWN_DT_TM').rename(columns={'DRAWN_DT_TM':'DRAWN_DTTM_Plas','COMPLETED_DT_TM':'COMPLETED_DT_TM_Plas'}),
        left_on='DRAWN_DTTM_WB',
        right_on='DRAWN_DTTM_Plas',
        by='PERSON_ID',
        direction='nearest'
    ).rename(columns={'RESULT_VALUE_NUMERIC':'Potassium WB','ACCESSION':'ACCESSION_WB','YRS_OLD_AT_COLLECTION':'Age_At_WB_Draw_Yrs'})

    res=res[~(res['Potassium Plas'].isna()|res['Potassium WB'].isna())].reset_index(drop=True)
    res['dt_wb_plas']=(res['DRAWN_DTTM_WB']-res['DRAWN_DTTM_Plas'])/np.timedelta64(1,'h')
    res=res.loc[abs(res['dt_wb_plas'])<threshold]
    res['LUO-H bins']=pd.cut(res['LUO-H'],[0,50,max(res['LUO-H'])],labels=['<50','>50'])
    if not ax:
        fig,ax=plt.subplots()
    
    sns.histplot(res['dt_wb_plas'],
                 ax=ax,
                 color='k',
                 binwidth=1,
                )
    sns.set_style("white")   
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Draw time interval (WB-Plas, hrs)')
    ax.set_xlim([-5,5])
    return(res)


def plot_blandaltman(df_plasma:pd.DataFrame,df_wb:pd.DataFrame,dt_threshold:float=2.,luoh_threshod:int=50):
    """Plots method comparison and bland altman plots"""
    df_paired=plot_dtwbplas(df_plasma,df_wb)
    filt_dt=(abs(df_paired['dt_wb_plas'])<dt_threshold)#&(res['dt_wb_plas']>0)
    filt_luoh=df_paired['LUO-H']<luoh_threshod
    fig,ax=plt.subplots(1,2)
    fig.set_size_inches(6,3)
    sns.set(font_scale=1)
    sns.set_style(style="white")
    slope, intercept, r_value, p_value, std_err = linregress(x=df_paired.loc[filt_dt&filt_luoh,'Potassium Plas'], y=df_paired.loc[filt_dt&filt_luoh,'Potassium WB'])
    ax[0].annotate(f'$y={slope:.3f}x{intercept:+.2f}$\n$r^2 = {r_value ** 2:.2f}$',
                xy=(.05, .95), xycoords=ax[0].transAxes, fontsize=8,
                color='black', backgroundcolor='#FFFFFF00', ha='left', va='top')
    sns.regplot(
        df_paired.loc[filt_dt&filt_luoh,'Potassium Plas'],
        df_paired.loc[filt_dt&filt_luoh,'Potassium WB'],
        scatter_kws={'alpha':0.5,
                     's':5,
                     'color':'gray',
                     'linewidth':0},
        line_kws={'color':'black',
                  'linewidth':1},        
        ax=ax[0]
    )
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_ylabel('WB K (mmol/L)')
    ax[0].set_xlabel('Plasma K (mmol/L)')
    ax[0].set_xlim([0,9])
    ax[0].set_ylim([0,9])
    ax[0].vlines(3.3,0,9,linestyle=':',color='black',linewidth=0.5)
    ax[0].vlines(5,0,9,linestyle=':',color='black',linewidth=0.5)
    ax[0].hlines(3.3,0,9,linestyle=':',color='black',linewidth=0.5)
    ax[0].hlines(4.9,0,9,linestyle=':',color='black',linewidth=0.5)
    sns.scatterplot(
        df_paired.loc[filt_dt&filt_luoh,'Potassium Plas'],
        df_paired.loc[filt_dt&filt_luoh,'Potassium WB']-df_paired.loc[filt_dt&filt_luoh,'Potassium Plas'],
        s=5,
        alpha=0.5,
        ax=ax[1],
        color='gray'
    )
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].plot([0,9],[0,0],'k',alpha=0.5,linestyle=':',linewidth=1)
    ax[1].set_ylabel('WB K - Plasma K (mmol/L)')
    ax[1].set_xlabel('Plasma K (mmol/L)')
    print((df_paired.loc[filt_dt&filt_luoh,'Potassium WB']-df_paired.loc[filt_dt&filt_luoh,'Potassium Plas']).mean())

    plt.tight_layout()

    ax[1].set_xlim([0,9])
    ax[1].set_ylim([-5,5])
    return(df_paired.loc[filt_dt&filt_luoh,'Potassium WB']-df_paired.loc[filt_dt&filt_luoh,'Potassium Plas'])

def plot_confusionmatrix(df_plasma:pd.DataFrame,df_wb:pd.DataFrame,dt_threshold:float=2.0,luoh_threshold:int=50):
    """Plots confusion matrix for WB vs Plasma"""
    res=plot_dtwbplas(df_plasma,df_wb)
    reference_interval=[3.3,5]
    filt_dt=(abs(res['dt_wb_plas'])<dt_threshold)#&(res['dt_wb_plas']>0)
    filt_luoh=res['LUO-H']<luoh_threshold
    res=res.loc[filt_dt&filt_luoh]
    res.loc[res['Potassium Plas']<reference_interval[0],'Plasma']='Hypokalemia'
    res.loc[(reference_interval[0]<=res['Potassium Plas'])&(res['Potassium Plas']<reference_interval[1]),'Plasma']='Normokalemia'
    res.loc[res['Potassium Plas']>=reference_interval[1],'Plasma']='Hyperkalemia'

    res.loc[res['Potassium WB']<reference_interval[0],'WB']='Hypokalemia'
    res.loc[(reference_interval[0]<=res['Potassium WB'])&(res['Potassium WB']<reference_interval[1]),'WB']='Normokalemia'
    res.loc[res['Potassium WB']>=reference_interval[1],'WB']='Hyperkalemia'

    cm=res.groupby(['Plasma','WB']).agg(num_pairs=pd.NamedAgg('ACCESSION_WB',pd.Series.nunique)).unstack(0)
    cm.columns=cm.columns.droplevel()#rename(columns={col:col[1] for col in cm.columns})
    cm=cm.loc[['Hyperkalemia', 'Normokalemia', 'Hypokalemia'],['Hypokalemia', 'Normokalemia', 'Hyperkalemia']]
    total=cm.sum().sum()
    fig,ax=plt.subplots()
    g = sns.heatmap(cm.fillna(0),
                annot=True,
                fmt=".0f",
                cmap='gray',
                linewidth=0.5,
                cbar=False,
                ax=ax)
    for t in g.texts: t.set_text(f'{int(t.get_text())} ({100*int(t.get_text())/total:.1f}%)')
    ax.set_ylabel('WB K classification')
    ax.set_xlabel('Plasma K classification')

    res=cm.fillna(0).applymap(lambda X: '%.0f (%.1f%%)'%(X,100*X/cm.sum().sum()))
    return(res)


def filter_hemolyzed(df_plasma:pd.DataFrame,threshold_hi:float=50)->pd.DataFrame:
    '''Returns dataframe after filtering plasma specimen hemolyzed beyond threshold'''
    filt_luo=df_plasma['TASK_ASSAY']=='LUO-H'
    acc_not_hemolyzed=set(df_plasma.loc[(filt_luo)&(df_plasma['RESULT_VALUE_NUMERIC']<threshold_hi),'ACCESSION'])
    dff=df_plasma.loc[df_plasma['ACCESSION'].isin(acc_not_hemolyzed)]    
    return(dff)


def paired_plasma_k(df_plasma:pd.DataFrame,threshold_hi:float=50,dt_threshold:int=2)->pd.DataFrame:
    '''Returns dataframe with non-heolyzed plasma-plasma k pairs within threshold time in hours'''
    dff=filter_hemolyzed(df_plasma,threshold_hi)
    dff=dff.loc[dff['TASK_ASSAY']=='Potassium Plas']
    dff['COLLECTED_DT_TM']=dff['DRAWN_DT_TM']
    dff=dff[['ACCESSION','DRAWN_DT_TM','COLLECTED_DT_TM','PERSON_ID','RESULT_VALUE_NUMERIC','TASK_ASSAY']].sort_values(by='DRAWN_DT_TM',ascending=True)
    dff=pd.merge_asof(dff,dff,on='DRAWN_DT_TM',by='PERSON_ID',direction='nearest',allow_exact_matches=False,tolerance=pd.Timedelta(f'{dt_threshold}h') )
    dff=dff.loc[(~dff['RESULT_VALUE_NUMERIC_x'].isna())&(~dff['RESULT_VALUE_NUMERIC_y'].isna())]    
    dff=dff.loc[dff['ACCESSION_x'].isin(dff['ACCESSION_y'])]
    return(dff)

def plot_plasma_plasma_differences(df_plasma_plasma:pd.DataFrame,ax=False):
    '''Plots comparison of paired non-hemolyzed plasma k results'''    
    fig,ax=plt.subplots(1,2)
    fig.set_size_inches(6,3)
    slope, intercept, r_value, p_value, std_err = linregress(x=df_plasma_plasma['RESULT_VALUE_NUMERIC_x'], y=df_plasma_plasma['RESULT_VALUE_NUMERIC_y'])
    ax[0].annotate(f'$y={slope:.3f}x{intercept:+.2f}$\n$r^2 = {r_value ** 2:.2f}$',
                xy=(.05, .95), xycoords=ax[0].transAxes, fontsize=8,
                color='black', backgroundcolor='#FFFFFF00', ha='left', va='top')
    sns.regplot(
        df_plasma_plasma['RESULT_VALUE_NUMERIC_x'],
        df_plasma_plasma['RESULT_VALUE_NUMERIC_y'],
        scatter_kws={'alpha':0.5,
                        's':5,
                        'color':'gray',
                        'linewidth':0},
        line_kws={'color':'black',
                    'linewidth':1},        
        ax=ax[0]
    )
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_ylabel('Plasma K (mmol/L) Result 1')
    ax[0].set_xlabel('Plasma K (mmol/L) Result 2')
    ax[0].set_xlim([0,9])
    ax[0].set_ylim([0,9])
    ax[0].vlines(3.3,0,9,linestyle=':',color='black',linewidth=0.5)
    ax[0].vlines(5,0,9,linestyle=':',color='black',linewidth=0.5)
    ax[0].hlines(3.3,0,9,linestyle=':',color='black',linewidth=0.5)
    ax[0].hlines(4.9,0,9,linestyle=':',color='black',linewidth=0.5)

    sns.scatterplot(
        df_plasma_plasma[['RESULT_VALUE_NUMERIC_x','RESULT_VALUE_NUMERIC_y']].mean(axis=1),
        df_plasma_plasma['RESULT_VALUE_NUMERIC_x']-df_plasma_plasma['RESULT_VALUE_NUMERIC_y'],
        s=5,
        alpha=0.5,
        ax=ax[1],
        color='gray'
    )
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].plot([0,9],[0,0],'k',alpha=0.5,linestyle=':',linewidth=1)
    ax[1].set_ylabel('Delta Plasma K (mmol/L)')
    ax[1].set_xlabel('Mean Plasma K (mmol/L)')
    plt.tight_layout()

    ax[1].set_xlim([0,9])
    ax[1].set_ylim([-5,5])    
    return(df_plasma_plasma['RESULT_VALUE_NUMERIC_x']-df_plasma_plasma['RESULT_VALUE_NUMERIC_y'])




#%%
def main():
    # plasma
    df_plasma=import_data('./all_plasma_v3')
    df_plasma=process_data(df_plasma)
    df_plasma=add_groupers(df_plasma)
    
    # wb
    df_wb=import_data('./wb_v2')
    df_wb=process_data(df_wb)
    df_wb=add_groupers(df_wb)

    # make plots
    plot_volumes(df_plasma,df_wb)
    plot_locgroupvolumes(df_plasma,df_wb)
    plot_agegroupvolumes(df_plasma,df_wb)
    plot_luohcdf(df_plasma)
    plot_kcdf(df_plasma,dta='Potassium Plas')
    plot_kcdf(df_wb,dta='Potassium WB')
    plot_kvluoh(df_plasma)
    wp_diff=plot_blandaltman(df_plasma,df_wb)
    res=plot_confusionmatrix(df_plasma,df_wb)
    df_plasma_plasma=paired_plasma_k(df_plasma,threshold_hi=50,dt_threshold=2)
    pp_diff=plot_plasma_plasma_differences(df_plasma_plasma)
    return(df_plasma,df_wb,res,wp_diff,pp_diff)

if __name__=='__main__':
    df_plasma,df_wb,res,wp_diff,pp_diff=main()
    print(res)

# %%
fig,ax=plt.subplots()
fig.set_size_inches(5,3)
sns.ecdfplot(
    wp_diff,
    ax=ax,
    color='k',
    label='WB-Plasma'
)
sns.ecdfplot(
    pp_diff,
    ax=ax,
    color='gray',
    label='Plasma-Plasma'
)
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('Difference in K (mmol/L)')
ax.set_ylabel('Cumulative fraction')
ax.set_xlim([-5,5])
ax.set_ylim([-0.05,1.05])
ax.legend()
ax.vlines(0,0,1,linestyle=':',color='k',linewidth=0.5)
ax.vlines(wp_diff.mean(),0,1,linestyle=':',color='r',linewidth=0.5)
ax.vlines(pp_diff.mean(),0,1,linestyle=':',color='g',linewidth=0.5)


# %%


from scipy.stats import bootstrap

res_wp = bootstrap((np.array(wp_diff),), np.mean)
res_pp = bootstrap((np.array(pp_diff),), np.mean)

fig,ax=plt.subplots()
sns.histplot(
    res_wp.bootstrap_distribution,
    ax=ax,
    label='WB-Plasma',
    color='k'
)
sns.histplot(
    res_pp.bootstrap_distribution,
    ax=ax,
    label='Plasma-Plasma',
    color='gray'
)
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('Mean Difference in K (mmol/L)')
ax.set_ylabel('# Bootstrap samples')



fig,ax=plt.subplots()
ax.boxplot(
    [res_wp.bootstrap_distribution,res_pp.bootstrap_distribution],)
ax.hlines(0,0,3,linestyle=':',color='k',linewidth=0.5)
ax.set_xticklabels(['WB-Plasma','Plasma-Plasma'])
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Mean Difference in K (mmol/L)')

#%%
print(
    f'WB-Plasma: Mean difference: {np.mean(wp_diff):.2f}\n',
    f'WB-Plasma: STD difference: {np.std(wp_diff):.2f}\n',
    f'WB-Plasma: n: {len(wp_diff)}\n',
    f'WB-Plasma: Bootstrap 5-95% CI: {np.percentile(res_wp.bootstrap_distribution,5):.2f} - {np.percentile(res_wp.bootstrap_distribution,95):.2f}\n',
    f'WB-Plasma: 1th percentile: {np.percentile(wp_diff,1):.2f}\n',
    f'WB-Plasma: 5th percentile: {np.percentile(wp_diff,5):.2f}\n',
    f'WB-Plasma: 50th percentile: {np.percentile(wp_diff,50):.2f}\n', 
    f'WB-Plasma: 95th percentile: {np.percentile(wp_diff,95):.2f}\n'
    f'WB-Plasma: 99th percentile: {np.percentile(wp_diff,99):.2f}\n'
)

print(
    f'Plasma-Plasma: Mean difference: {np.mean(pp_diff):.2f}\n',
    f'Plasma-Plasma: STD difference: {np.std(pp_diff):.2f}\n',
    f'Plasma-Plasma: n: {len(pp_diff)}\n',
    f'Plasma-Plasma: Bootstrap 5-95% CI: {np.percentile(res_pp.bootstrap_distribution,5):.2f} - {np.percentile(res_pp.bootstrap_distribution,95):.2f}\n',
    f'Plasma-Plasma: 1th percentile: {np.percentile(pp_diff,1):.2f}\n',
    f'Plasma-Plasma: 5th percentile: {np.percentile(pp_diff,5):.2f}\n',
    f'Plasma-Plasma: 50th percentile: {np.percentile(pp_diff,50):.2f}\n',
    f'Plasma-Plasma: 95th percentile: {np.percentile(pp_diff,95):.2f}\n'
    f'Plasma-Plasma: 99th percentile: {np.percentile(pp_diff,99):.2f}\n'
)
