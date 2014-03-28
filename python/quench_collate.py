import re
import csv
import json
import numpy as np
import operator
import pandas as pd
from pymongo import MongoClient
from astropy.io import fits as pyfits

'''
To create the full collated data for GZ: Quench:

>>> import quench_collate as qc
>>> listcoll = qc.collate_answers()
>>> qc.write_fits(listcoll)
>>> qc.write_csv(listcoll)

- Kyle Willett (willett@physics.umn.edu)

29 Aug 2013 - corrected the decision tree, moving question 8 outside question 2 requirement.
26 Mar 2014 - added pandas dataframe support; easier ability to remove duplicate classifications. 
                Would like to reformat the whole rest of code to use dataframes when I have time.
'''

quenchdir = '/Users/willettk/Astronomy/Research/GalaxyZoo/quench/'
writedir = '%s/quench_for_volunteers' % quenchdir
filename = '2013-08-29_galaxy_zoo_starburst_classifications.csv'

def load_data():

    f = open(quenchdir+filename,'rb')
    a = f.readlines()
    f.close()

    return a

def load_df():

    df = pd.DataFrame.from_csv('%s/%s' % (quenchdir,filename))

    return df

def data_reader():

    f = open(quenchdir+filename,'rb')
    reader = csv.reader(f)
    headers = reader.next()
    column = {}
    
    for h in headers:
        column[h] = []

    for row in reader:
        for h,v in zip(headers,row):
            column[h].append(v)

    return column

def unique_subjects():

    column = data_reader()

    subjects = set(column['subject_id'])

    sdss = []
    for sub in subjects:
        sdss.append(column['sdss_id'][column['subject_id'].index(sub)])

    return subjects,sdss

def unique_sdss():

    column = data_reader()

    sdss = set(column['sdss_id'])

    subjects = []
    for s in sdss:
        subjects.append(column['subject_id'][column['sdss_id'].index(s)])

    return subjects,sdss

def colldict():

    collated = {}
    collated["mini_project-0"] =  {"a-0":0,"a-1":0 ,"a-2":0}
    collated["mini_project-1"] =  {"a-0":0,"a-1":0 ,"a-2":0}
    collated["mini_project-2"] =  {"a-0":0,"a-1":0}
    collated["mini_project-3"] =  {"a-0":0,"a-1":0}
    collated["mini_project-4"] =  {"a-0":0,"a-1":0}
    collated["mini_project-5"] =  {"a-0":0,"a-1":0}
    collated["mini_project-6"] =  {"a-0":0,"a-1":0 ,"a-2":0}
    collated["mini_project-7"] =  {"a-0":0,"a-1":0 ,"a-2":0}
    collated["mini_project-8"] =  {"a-0":0,"a-1":0 ,"a-2":0}
    collated["mini_project-9"] =  {"a-0":0,"a-1":0 ,"a-2":0 ,"a-3":0}
    collated["mini_project-10"] = {"a-0":0,"a-1":0}
    collated["mini_project-11"] = {"a-0":0,"a-1":0}

    return collated

def collate_answers(bysdss=False,drop_duplicates=True):

    # Decide whether sorting answers by SDSS ID or by Zooniverse subject ID
    if bysdss:
        subjects,sdss = unique_sdss()
    else:
        subjects,sdss = unique_subjects()

    # Load raw classifications from csv file
    #rows = load_data()
    df = load_df()

    # Empty list
    listcoll = []

    # Loop over all unique subjects
    for subject_id,sdss_id in zip(subjects,sdss):
        # Empty dictionary of tasks
        collated = colldict()
        if bysdss:
            keyid,keyind = sdss_id,'sdss_id'
        else:
            keyid,keyind = subject_id,'subject_id'

        mr = df[(df[keyind] == keyid)].copy()

        # Sort by classification date, since we only keep the last one
        mr.sort(['created_at'])

        if drop_duplicates:
            # Users who weren't logged in have no user_id. Assume that they actually were
            #   individual users, so we'll keep those classifications.
            anonymous_users = mr['user_id'].isnull()
            unique_logged = mr[~anonymous_users].drop_duplicates('user_id', take_last=True)
            if anonymous_users.sum() > 0:
                unique_logged.append(mr[anonymous_users])
            mr = unique_logged

        for k in mr.columns[4:]:
            tc = mr.copy()[k].dropna()
            vc = tc.value_counts()
            if len(vc) > 0:
                for vk in vc.keys():
                    collated[k][vk] += vc[vk]

        # Append the IDs and collated dictionary with task counts to list
        listcoll.append((subject_id,sdss_id,collated))

    return listcoll

def collated_df(listcoll):

    listseries = []
    for l in listcoll:
        listseries.append(pd.Series((l[0],l[1],l[2]),index=['subject_id','sdss_id','collated']))

    dfcoll = pd.DataFrame(listseries)

    return dfcoll

def dict_to_str(gzq_dict):

    dictstr = "{'gzquench':['mini_project-0':%s,'mini_project-1':%s,'mini_project-2':%s,'mini_project-3':%s,'mini_project-4':%s,'mini_project-5':%s,'mini_project-6':%s,'mini_project-7':%s,'mini_project-8':%s,'mini_project-9':%s,'mini_project-10':%s,'mini_project-11':%s]}" % (gzq_dict['mini_project-0'],gzq_dict['mini_project-1'],gzq_dict['mini_project-2'],gzq_dict['mini_project-3'],gzq_dict['mini_project-4'],gzq_dict['mini_project-5'],gzq_dict['mini_project-6'],gzq_dict['mini_project-7'],gzq_dict['mini_project-8'],gzq_dict['mini_project-9'],gzq_dict['mini_project-10'],gzq_dict['mini_project-11'])

    dictstr_nospace = dictstr.translate(None,' ')

    return dictstr_nospace
    
def write_csv(listcoll,bysdss=False):

    filestub = '_bysdss' if bysdss else '_bysubject'
    writefilename = 'consensus/gzquench_consensus%s.csv' % filestub

    f = open(quenchdir+writefilename,'wb')

    # Write the column headers
    f.write('subject_id,sdss_id,vote_total,most_common_path,votes\n')
    for l in listcoll:

        votetotal = np.sum(l[2]['mini_project-0'].values())
        mcp = quench_tree(l[2])
        splititem = (str(l[0]),str(l[1]),str(votetotal),mcp,json.dumps(l[2])+'\n')
        writeitem = '\t'.join(splititem)

        f.write(writeitem)

    f.close()

    return None

def write_csv_nojson(listcoll):

    writefilename = 'data_forvolunteers/gzquench_consensus_nojson.csv'

    tasks = ['mini_project-%i' % i for i in np.arange(12)] 

    f = open(quenchdir+writefilename,'wb')

    # Write the column headers
    header_1 = 'sdss_id,vote_total,most_common_path'
    responses = 't00_a00,t00_a01,t00_a02,t01_a00,t01_a01,t01_a02,t02_a00,t02_a01,t03_a00,t03_a01,t04_a00,t04_a01,t05_a00,t05_a01,t06_a00,t06_a01,t06_a02,t07_a00,t07_a01,t07_a02,t08_a00,t08_a01,t08_a02,t09_a00,t09_a01,t09_a02,t09_a03,t10_a00,t10_a01,t11_a00,t11_a01'

    metadata_keys = ( u'control', u'fiberid', u'plateid', u'mjd', u'redshift', u'redshift_err', u'v_disp', u'v_disp_err', u'log_mass', u'u_absmag', u'umag', u'g_absmag', u'gmag', u'r_absmag', u'rmag', u'i_absmag', u'imag', u'z_absmag', u'zmag',u'abs_r', u'petro_r50', u'd4000', u'd4000_err', u'halpha_flux', u'halpha_flux_err', u'hbeta_flux', u'hbeta_flux_err', u'nii_flux', u'nii_flux_err', u'oii_flux', u'oii_flux_err', u'oiii_flux', u'oiii_flux_err', u'nad_abs_flux', u'nad_abs_flux_err', u'arm_tightness', u'center_bar', u'central_bulge', u'central_bulge_prominence', u'clumps', u'disk_edge', u'how_round', u'merging', u'smooth', u'spiral_arms', u'symmetrical')

    header_responses_count = ','.join(['%s_count' % r for r in responses.rsplit(',')])
    header_responses_frac = ','.join(['%s_fraction' % r for r in responses.rsplit(',')])
    header_responses_metadata = ','.join(['%s' % r for r in metadata_keys])

    f.write('%s,%s,%s,%s\n' % (header_1,header_responses_count,header_responses_frac,header_responses_metadata))

    control_file = '%s/quench_data/control_updated.json' % quenchdir
    sample_file = '%s/quench_data/sample_updated.json' % quenchdir

    cf = open(control_file)
    control = json.load(cf)
    cf.close()

    sf = open(sample_file)
    sample = json.load(sf)
    sf.close()

    control_ids = [x[u'id'] for x in control]
    sample_ids = [x[u'id'] for x in sample]
    missing = 0

    for l in listcoll:

        votes = l[2]
        votetotal = np.sum(votes['mini_project-0'].values())
        mcp = quench_tree(votes)

        # Vote total for each task

        tlist = [(t,np.sum(votes['mini_project-%i' % t].values())) for t in np.arange(12)]
        tdict = {key: value for (key, value) in tlist}

        countlist = []
        fraclist = []
        for r in responses.rsplit(','):
            response_split = r.rsplit('_')
            task_no = int(response_split[0][1:])
            resp_no = int(response_split[1][1:])

            key_task = 'mini_project-%i' % task_no
            key_response = 'a-%i' % resp_no

            n_response = votes[key_task][key_response]
            n_task = tdict[task_no]

            countlist.append(n_response)
            fraclist.append('%.3f' % (n_response/float(n_task))) if n_task > 0 else fraclist.append('0.')

        # Find metadata from JSON file

        if l[0] in control_ids or l[0] in sample_ids:
            if l[0] in control_ids:
                idx = control_ids.index(l[0])
                msource = control
            else:
                idx = sample_ids.index(l[0])
                msource = sample

            m = []
            for k in metadata_keys:
                if k in msource[idx]['metadata'].keys():
                    m.append(msource[idx]['metadata'][k])
                else:
                    m.append(None)

            # Only write to file for objects with matching metadata 

            splititem = (str(l[1]),str(votetotal),mcp,','.join([str(rr) for rr in countlist]),','.join([str(rr) for rr in fraclist]),','.join([str(rr) for rr in m])+'\n')
            writeitem = ',\t'.join(splititem)

            f.write(writeitem)

        else:
            missing += 1

    f.close()

    print 'Missing %i' % missing

    return None


def write_fits(listcoll,bysdss=False):

    filestub = '_bysdss' if bysdss else '_bysubject'
    writefilename = 'consensus/gzquench_consensus%s.fits' % filestub

    subject_id = []
    sdss_id = []
    total_votes = []
    mostcommonpath = []
    t00_a00_count,t00_a01_count,t00_a02_count = [],[],[]
    t00_a00_fraction,t00_a01_fraction,t00_a02_fraction = [],[],[]
    t01_a00_count,t01_a01_count,t01_a02_count = [],[],[]
    t01_a00_fraction,t01_a01_fraction,t01_a02_fraction = [],[],[]
    t02_a00_count,t02_a01_count= [],[]
    t02_a00_fraction,t02_a01_fraction= [],[]
    t03_a00_count,t03_a01_count= [],[]
    t03_a00_fraction,t03_a01_fraction= [],[]
    t04_a00_count,t04_a01_count= [],[]
    t04_a00_fraction,t04_a01_fraction= [],[]
    t05_a00_count,t05_a01_count= [],[]
    t05_a00_fraction,t05_a01_fraction= [],[]
    t06_a00_count,t06_a01_count,t06_a02_count = [],[],[]
    t06_a00_fraction,t06_a01_fraction,t06_a02_fraction = [],[],[]
    t07_a00_count,t07_a01_count,t07_a02_count = [],[],[]
    t07_a00_fraction,t07_a01_fraction,t07_a02_fraction = [],[],[]
    t08_a00_count,t08_a01_count,t08_a02_count = [],[],[]
    t08_a00_fraction,t08_a01_fraction,t08_a02_fraction = [],[],[]
    t09_a00_count,t09_a01_count,t09_a02_count,t09_a03_count = [],[],[],[]
    t09_a00_fraction,t09_a01_fraction,t09_a02_fraction,t09_a03_fraction = [],[],[],[]
    t10_a00_count,t10_a01_count= [],[]
    t10_a00_fraction,t10_a01_fraction= [],[]
    t11_a00_count,t11_a01_count= [],[]
    t11_a00_fraction,t11_a01_fraction= [],[]

    for l in listcoll:
        subject_id.append(l[0])
        sdss_id.append(l[1])
        total_votes.append(np.sum(l[2]['mini_project-0'].values()))
        mostcommonpath.append(quench_tree(l[2]))
        t00_count = np.sum(l[2]['mini_project-0'].values()).astype(float)
        t01_count = np.sum(l[2]['mini_project-1'].values()).astype(float)
        t02_count = np.sum(l[2]['mini_project-2'].values()).astype(float)
        t03_count = np.sum(l[2]['mini_project-3'].values()).astype(float)
        t04_count = np.sum(l[2]['mini_project-4'].values()).astype(float)
        t05_count = np.sum(l[2]['mini_project-5'].values()).astype(float)
        t06_count = np.sum(l[2]['mini_project-6'].values()).astype(float)
        t07_count = np.sum(l[2]['mini_project-7'].values()).astype(float)
        t08_count = np.sum(l[2]['mini_project-8'].values()).astype(float)
        t09_count = np.sum(l[2]['mini_project-9'].values()).astype(float)
        t10_count = np.sum(l[2]['mini_project-10'].values()).astype(float)
        t11_count = np.sum(l[2]['mini_project-11'].values()).astype(float)
        t00_a00_count.append(l[2]['mini_project-0']['a-0'])
        t00_a01_count.append(l[2]['mini_project-0']['a-1'])
        t00_a02_count.append(l[2]['mini_project-0']['a-2'])
        t00_a00_fraction.append(l[2]['mini_project-0']['a-0']/t00_count if t00_count > 0. else 0.)
        t00_a01_fraction.append(l[2]['mini_project-0']['a-1']/t00_count if t00_count > 0. else 0.)
        t00_a02_fraction.append(l[2]['mini_project-0']['a-2']/t00_count if t00_count > 0. else 0.)
        t01_a00_count.append(l[2]['mini_project-1']['a-0'])
        t01_a01_count.append(l[2]['mini_project-1']['a-1'])
        t01_a02_count.append(l[2]['mini_project-1']['a-2'])
        t01_a00_fraction.append(l[2]['mini_project-1']['a-0']/t01_count if t01_count > 0. else 0.)
        t01_a01_fraction.append(l[2]['mini_project-1']['a-1']/t01_count if t01_count > 0. else 0.)
        t01_a02_fraction.append(l[2]['mini_project-1']['a-2']/t01_count if t01_count > 0. else 0.)
        t02_a00_count.append(l[2]['mini_project-2']['a-0'])
        t02_a01_count.append(l[2]['mini_project-2']['a-1'])
        t02_a00_fraction.append(l[2]['mini_project-2']['a-0']/t02_count if t02_count > 0. else 0.)
        t02_a01_fraction.append(l[2]['mini_project-2']['a-1']/t02_count if t02_count > 0. else 0.)
        t03_a00_count.append(l[2]['mini_project-3']['a-0'])
        t03_a01_count.append(l[2]['mini_project-3']['a-1'])
        t03_a00_fraction.append(l[2]['mini_project-3']['a-0']/t03_count if t03_count > 0. else 0.)
        t03_a01_fraction.append(l[2]['mini_project-3']['a-1']/t03_count if t03_count > 0. else 0.)
        t04_a00_count.append(l[2]['mini_project-4']['a-0'])
        t04_a01_count.append(l[2]['mini_project-4']['a-1'])
        t04_a00_fraction.append(l[2]['mini_project-4']['a-0']/t04_count if t04_count > 0. else 0.)
        t04_a01_fraction.append(l[2]['mini_project-4']['a-1']/t04_count if t04_count > 0. else 0.)
        t05_a00_count.append(l[2]['mini_project-5']['a-0'])
        t05_a01_count.append(l[2]['mini_project-5']['a-1'])
        t05_a00_fraction.append(l[2]['mini_project-5']['a-0']/t05_count if t05_count > 0. else 0.)
        t05_a01_fraction.append(l[2]['mini_project-5']['a-1']/t05_count if t05_count > 0. else 0.)
        t06_a00_count.append(l[2]['mini_project-6']['a-0'])
        t06_a01_count.append(l[2]['mini_project-6']['a-1'])
        t06_a02_count.append(l[2]['mini_project-6']['a-2'])
        t06_a00_fraction.append(l[2]['mini_project-6']['a-0']/t06_count if t06_count > 0. else 0.)
        t06_a01_fraction.append(l[2]['mini_project-6']['a-1']/t06_count if t06_count > 0. else 0.)
        t06_a02_fraction.append(l[2]['mini_project-6']['a-2']/t06_count if t06_count > 0. else 0.)
        t07_a00_count.append(l[2]['mini_project-7']['a-0'])
        t07_a01_count.append(l[2]['mini_project-7']['a-1'])
        t07_a02_count.append(l[2]['mini_project-7']['a-2'])
        t07_a00_fraction.append(l[2]['mini_project-7']['a-0']/t07_count if t07_count > 0. else 0.)
        t07_a01_fraction.append(l[2]['mini_project-7']['a-1']/t07_count if t07_count > 0. else 0.)
        t07_a02_fraction.append(l[2]['mini_project-7']['a-2']/t07_count if t07_count > 0. else 0.)
        t08_a00_count.append(l[2]['mini_project-8']['a-0'])
        t08_a01_count.append(l[2]['mini_project-8']['a-1'])
        t08_a02_count.append(l[2]['mini_project-8']['a-2'])
        t08_a00_fraction.append(l[2]['mini_project-8']['a-0']/t08_count if t08_count > 0. else 0.)
        t08_a01_fraction.append(l[2]['mini_project-8']['a-1']/t08_count if t08_count > 0. else 0.)
        t08_a02_fraction.append(l[2]['mini_project-8']['a-2']/t08_count if t08_count > 0. else 0.)
        t09_a00_count.append(l[2]['mini_project-9']['a-0'])
        t09_a01_count.append(l[2]['mini_project-9']['a-1'])
        t09_a02_count.append(l[2]['mini_project-9']['a-2'])
        t09_a03_count.append(l[2]['mini_project-9']['a-3'])
        t09_a00_fraction.append(l[2]['mini_project-9']['a-0']/t09_count if t09_count > 0. else 0.)
        t09_a01_fraction.append(l[2]['mini_project-9']['a-1']/t09_count if t09_count > 0. else 0.)
        t09_a02_fraction.append(l[2]['mini_project-9']['a-2']/t09_count if t09_count > 0. else 0.)
        t09_a03_fraction.append(l[2]['mini_project-9']['a-3']/t09_count if t09_count > 0. else 0.)
        t10_a00_count.append(l[2]['mini_project-10']['a-0'])
        t10_a01_count.append(l[2]['mini_project-10']['a-1'])
        t10_a00_fraction.append(l[2]['mini_project-10']['a-0']/t10_count if t10_count > 0. else 0.)
        t10_a01_fraction.append(l[2]['mini_project-10']['a-1']/t10_count if t10_count > 0. else 0.)
        t11_a00_count.append(l[2]['mini_project-11']['a-0'])
        t11_a01_count.append(l[2]['mini_project-11']['a-1'])
        t11_a00_fraction.append(l[2]['mini_project-11']['a-0']/t11_count if t11_count > 0. else 0.)
        t11_a01_fraction.append(l[2]['mini_project-11']['a-1']/t11_count if t11_count > 0. else 0.)

    col_subject_id = pyfits.Column(name = 'subject_id', format='A24', array=subject_id)
    col_sdss_id = pyfits.Column(name = 'sdss_id', format='K', array=sdss_id)
    col_total_votes = pyfits.Column(name = 'total_votes', format='I4', array=total_votes)
    col_mostcommonpath = pyfits.Column(name = 'most_common_path', format='A100', array=mostcommonpath)
    col_t00_a00_count = pyfits.Column(name = 't00_a00_count', format='I4', array=t00_a00_count)
    col_t00_a01_count = pyfits.Column(name = 't00_a01_count', format='I4', array=t00_a01_count)
    col_t00_a02_count = pyfits.Column(name = 't00_a02_count', format='I4', array=t00_a02_count)
    col_t00_a00_fraction = pyfits.Column(name = 't00_a00_fraction', format='E5.3', array=t00_a00_fraction)
    col_t00_a01_fraction = pyfits.Column(name = 't00_a01_fraction', format='E5.3', array=t00_a01_fraction)
    col_t00_a02_fraction = pyfits.Column(name = 't00_a02_fraction', format='E5.3', array=t00_a02_fraction)
    col_t01_a00_count = pyfits.Column(name = 't01_a00_count', format='I4', array=t01_a00_count)
    col_t01_a01_count = pyfits.Column(name = 't01_a01_count', format='I4', array=t01_a01_count)
    col_t01_a02_count = pyfits.Column(name = 't01_a02_count', format='I4', array=t01_a02_count)
    col_t01_a00_fraction = pyfits.Column(name = 't01_a00_fraction', format='E5.3', array=t01_a00_fraction)
    col_t01_a01_fraction = pyfits.Column(name = 't01_a01_fraction', format='E5.3', array=t01_a01_fraction)
    col_t01_a02_fraction = pyfits.Column(name = 't01_a02_fraction', format='E5.3', array=t01_a02_fraction)
    col_t02_a00_count = pyfits.Column(name = 't02_a00_count', format='I4', array=t02_a00_count)
    col_t02_a01_count = pyfits.Column(name = 't02_a01_count', format='I4', array=t02_a01_count)
    col_t02_a00_fraction = pyfits.Column(name = 't02_a00_fraction', format='E5.3', array=t02_a00_fraction)
    col_t02_a01_fraction = pyfits.Column(name = 't02_a01_fraction', format='E5.3', array=t02_a01_fraction)
    col_t03_a00_count = pyfits.Column(name = 't03_a00_count', format='I4', array=t03_a00_count)
    col_t03_a01_count = pyfits.Column(name = 't03_a01_count', format='I4', array=t03_a01_count)
    col_t03_a00_fraction = pyfits.Column(name = 't03_a00_fraction', format='E5.3', array=t03_a00_fraction)
    col_t03_a01_fraction = pyfits.Column(name = 't03_a01_fraction', format='E5.3', array=t03_a01_fraction)
    col_t04_a00_count = pyfits.Column(name = 't04_a00_count', format='I4', array=t04_a00_count)
    col_t04_a01_count = pyfits.Column(name = 't04_a01_count', format='I4', array=t04_a01_count)
    col_t04_a00_fraction = pyfits.Column(name = 't04_a00_fraction', format='E5.3', array=t04_a00_fraction)
    col_t04_a01_fraction = pyfits.Column(name = 't04_a01_fraction', format='E5.3', array=t04_a01_fraction)
    col_t05_a00_count = pyfits.Column(name = 't05_a00_count', format='I4', array=t05_a00_count)
    col_t05_a01_count = pyfits.Column(name = 't05_a01_count', format='I4', array=t05_a01_count)
    col_t05_a00_fraction = pyfits.Column(name = 't05_a00_fraction', format='E5.3', array=t05_a00_fraction)
    col_t05_a01_fraction = pyfits.Column(name = 't05_a01_fraction', format='E5.3', array=t05_a01_fraction)
    col_t06_a00_count = pyfits.Column(name = 't06_a00_count', format='I4', array=t06_a00_count)
    col_t06_a01_count = pyfits.Column(name = 't06_a01_count', format='I4', array=t06_a01_count)
    col_t06_a02_count = pyfits.Column(name = 't06_a02_count', format='I4', array=t06_a02_count)
    col_t06_a00_fraction = pyfits.Column(name = 't06_a00_fraction', format='E5.3', array=t06_a00_fraction)
    col_t06_a01_fraction = pyfits.Column(name = 't06_a01_fraction', format='E5.3', array=t06_a01_fraction)
    col_t06_a02_fraction = pyfits.Column(name = 't06_a02_fraction', format='E5.3', array=t06_a02_fraction)
    col_t07_a00_count = pyfits.Column(name = 't07_a00_count', format='I4', array=t07_a00_count)
    col_t07_a01_count = pyfits.Column(name = 't07_a01_count', format='I4', array=t07_a01_count)
    col_t07_a02_count = pyfits.Column(name = 't07_a02_count', format='I4', array=t07_a02_count)
    col_t07_a00_fraction = pyfits.Column(name = 't07_a00_fraction', format='E5.3', array=t07_a00_fraction)
    col_t07_a01_fraction = pyfits.Column(name = 't07_a01_fraction', format='E5.3', array=t07_a01_fraction)
    col_t07_a02_fraction = pyfits.Column(name = 't07_a02_fraction', format='E5.3', array=t07_a02_fraction)
    col_t08_a00_count = pyfits.Column(name = 't08_a00_count', format='I4', array=t08_a00_count)
    col_t08_a01_count = pyfits.Column(name = 't08_a01_count', format='I4', array=t08_a01_count)
    col_t08_a02_count = pyfits.Column(name = 't08_a02_count', format='I4', array=t08_a02_count)
    col_t08_a00_fraction = pyfits.Column(name = 't08_a00_fraction', format='E5.3', array=t08_a00_fraction)
    col_t08_a01_fraction = pyfits.Column(name = 't08_a01_fraction', format='E5.3', array=t08_a01_fraction)
    col_t08_a02_fraction = pyfits.Column(name = 't08_a02_fraction', format='E5.3', array=t08_a02_fraction)
    col_t09_a00_count = pyfits.Column(name = 't09_a00_count', format='I4', array=t09_a00_count)
    col_t09_a01_count = pyfits.Column(name = 't09_a01_count', format='I4', array=t09_a01_count)
    col_t09_a02_count = pyfits.Column(name = 't09_a02_count', format='I4', array=t09_a02_count)
    col_t09_a03_count = pyfits.Column(name = 't09_a03_count', format='I4', array=t09_a03_count)
    col_t09_a00_fraction = pyfits.Column(name = 't09_a00_fraction', format='E5.3', array=t09_a00_fraction)
    col_t09_a01_fraction = pyfits.Column(name = 't09_a01_fraction', format='E5.3', array=t09_a01_fraction)
    col_t09_a02_fraction = pyfits.Column(name = 't09_a02_fraction', format='E5.3', array=t09_a02_fraction)
    col_t09_a03_fraction = pyfits.Column(name = 't09_a03_fraction', format='E5.3', array=t09_a03_fraction)
    col_t10_a00_count = pyfits.Column(name = 't10_a00_count', format='I4', array=t10_a00_count)
    col_t10_a01_count = pyfits.Column(name = 't10_a01_count', format='I4', array=t10_a01_count)
    col_t10_a00_fraction = pyfits.Column(name = 't10_a00_fraction', format='E5.3', array=t10_a00_fraction)
    col_t10_a01_fraction = pyfits.Column(name = 't10_a01_fraction', format='E5.3', array=t10_a01_fraction)
    col_t11_a00_count = pyfits.Column(name = 't11_a00_count', format='I4', array=t11_a00_count)
    col_t11_a01_count = pyfits.Column(name = 't11_a01_count', format='I4', array=t11_a01_count)
    col_t11_a00_fraction = pyfits.Column(name = 't11_a00_fraction', format='E5.3', array=t11_a00_fraction)
    col_t11_a01_fraction = pyfits.Column(name = 't11_a01_fraction', format='E5.3', array=t11_a01_fraction)

    primary_hdu = pyfits.PrimaryHDU()
    hdulist = pyfits.HDUList([primary_hdu])
        
    tb1_hdu = pyfits.new_table([\
                                col_subject_id,  
                                col_sdss_id,  
                                col_total_votes,
                                col_mostcommonpath,
                                col_t00_a00_count,
                                col_t00_a01_count,
                                col_t00_a02_count,
                                col_t00_a00_fraction,
                                col_t00_a01_fraction,
                                col_t00_a02_fraction,
                                col_t01_a00_count,
                                col_t01_a01_count,
                                col_t01_a02_count,
                                col_t01_a00_fraction,
                                col_t01_a01_fraction,
                                col_t01_a02_fraction,
                                col_t02_a00_count,
                                col_t02_a01_count,
                                col_t02_a00_fraction,
                                col_t02_a01_fraction,
                                col_t03_a00_count,
                                col_t03_a01_count,
                                col_t03_a00_fraction,
                                col_t03_a01_fraction,
                                col_t04_a00_count,
                                col_t04_a01_count,
                                col_t04_a00_fraction,
                                col_t04_a01_fraction,
                                col_t05_a00_count,
                                col_t05_a01_count,
                                col_t05_a00_fraction,
                                col_t05_a01_fraction,
                                col_t06_a00_count,
                                col_t06_a01_count,
                                col_t06_a02_count,
                                col_t06_a00_fraction,
                                col_t06_a01_fraction,
                                col_t06_a02_fraction,
                                col_t07_a00_count,
                                col_t07_a01_count,
                                col_t07_a02_count,
                                col_t07_a00_fraction,
                                col_t07_a01_fraction,
                                col_t07_a02_fraction,
                                col_t08_a00_count,
                                col_t08_a01_count,
                                col_t08_a02_count,
                                col_t08_a00_fraction,
                                col_t08_a01_fraction,
                                col_t08_a02_fraction,
                                col_t09_a00_count,
                                col_t09_a01_count,
                                col_t09_a02_count,
                                col_t09_a03_count,
                                col_t09_a00_fraction,
                                col_t09_a01_fraction,
                                col_t09_a02_fraction,
                                col_t09_a03_fraction,
                                col_t10_a00_count,
                                col_t10_a01_count,
                                col_t10_a00_fraction,
                                col_t10_a01_fraction,
                                col_t11_a00_count,
                                col_t11_a01_count,
                                col_t11_a00_fraction,
                                col_t11_a01_fraction,])

    tb1_hdu.name = 'GZQUENCH'
    
    hdulist.append(tb1_hdu)
    hdulist.writeto(quenchdir+writefilename,clobber=True)    

    return None

def max_item(jdict):

    mi = max(jdict.iteritems(), key=operator.itemgetter(1))[0]

    return mi

def quench_tree(gal):

    keys = gal.keys()
    assert 'mini_project-0' in keys, \
        'Cannot find mini_project-0 in keys'

    char = ''

    # First answer

    s0_max = max_item(gal['mini_project-0'])
    char += 's0%s;' % re.sub('-','',str(s0_max))

    # Star/artifact

    if s0_max != 'a-2':

        # Smooth galaxies

        if s0_max == 'a-0':

            s1_max = max_item(gal['mini_project-1'])
            char += 's1%s;' % re.sub('-','',str(s1_max))

        # Features/disk

        if s0_max == 'a-1':
            s2_max = max_item(gal['mini_project-2'])
            char += 's2%s;' % re.sub('-','',str(s2_max))
            # Edge-on disk
            if s2_max == 'a-0':
                # Edge on bulge
                s3_max = max_item(gal['mini_project-8'])
                char += 's3%s;' % re.sub('-','',str(s3_max))
            # Not edge-on disk
            else:
                # Bar
                s4_max = max_item(gal['mini_project-4'])
                char += 's4%s;' % re.sub('-','',str(s4_max))
                # Spiral
                s5_max = max_item(gal['mini_project-5'])
                char += 's5%s;' % re.sub('-','',str(s5_max))
                if s5_max == 'a-0':
                    # Arm tightness
                    s6_max = max_item(gal['mini_project-6'])
                    char += 's6%s;' % re.sub('-','',str(s6_max))
                # Bulge prominence
                s7_max = max_item(gal['mini_project-7'])
                char += 's7%s;' % re.sub('-','',str(s7_max))

        # Off-center bright clumps?
        s8_max = max_item(gal['mini_project-8'])
        char += 's8%s;' % re.sub('-','',str(s8_max))
        # Merging or tidal debris?
        s9_max = max_item(gal['mini_project-9'])
        char += 's9%s;' % re.sub('-','',str(s9_max))
        # Symmetrical?
        s10_max = max_item(gal['mini_project-10'])
        char += 's10%s;' % re.sub('-','',str(s10_max))
        # Discuss object?
        s11_max = max_item(gal['mini_project-11'])
        char += 's11%s;' % re.sub('-','',str(s11_max))

    return char

def find_duplicates():

    cols = data_reader()
    subjects,sdss = unique_subjects()

    for s in subjects:
        smatch=[]  
        for idx,si in enumerate(cols['subject_id']):
            if si == s:
                smatch.append(idx)
        sdss_ids = [cols['sdss_id'][i] for i in smatch]
        if len(set(sdss_ids)) > 1:
            print s,set(sdss_ids)

    return None

def mergers_new(listcoll):

    merger_dict = {'a-0':'Merging','a-1':'Tidal debris','a-2':'Both','a-3':'Neither','a-4':'Disturbed'}

    new_responses = []
    f = open('/Users/willettk/Astronomy/Research/GalaxyZoo/quench/new_merger_classes.txt','wb')
    for l in listcoll:
        responses = l[2]
        max_answer = max_item(responses['mini_project-9'])
        if max_answer == 'a-3':
            if (responses['mini_project-9']['a-0'] + responses['mini_project-9']['a-1'] + responses['mini_project-9']['a-2']) > responses['mini_project-9']['a-3']:
                true_response = 'a-4'
            else:
                true_response = max_answer
        else:
            true_response = max_answer

        new_responses.append(true_response)

        # Write to file

        f.write('%s\t%s\t%s\t%s\n' % (l[0],l[1],true_response,merger_dict[true_response]))

    f.close()

    return new_responses

def count_duplicate_user_classifications():

    subjects,sdss = unique_sdss()

    rows = load_data()

    dupcount = []

    for idx,sdss_id in enumerate(list(sdss)):
        if idx % 100 == 0:
            print idx
        user_ids = []
        uc = 0
        for row in rows[1:]:
            row_sdss_id = row.split(',')[2]
            if sdss_id == row_sdss_id:
                user_ids.append(row.split(',')[3])
                uc += 1
            unique_users = set(user_ids)
        bu = user_ids.count('')
        uu = len(unique_users)
        if bu > 0:
            dupcount.append(uc - uu - bu + 1)
        else:
            dupcount.append(uc - uu)

    return dupcount

def load_mongo_data():

    # Connect to Mongo database
    # Make sure to run mongorestore /path/to/database to restore the updated files
    # mongod client must be running locally
    
    client = MongoClient('localhost', 27017)
    db = client['ouroboros'] 
    
    subjects = db['galaxy_zoo_starburst_subjects'] 		# subjects = images
    classifications = db['galaxy_zoo_starburst_classifications']	# classifications = classifications of each subject per user
    users = db['galaxy_zoo_starburst_users']	# volunteers doing each classification (can be anonymous)

    return subjects,classifications,users

def duplicate_zooniverse_ids(dfcoll):

    subjects,classifications,users = load_mongo_data()
    
    dups = dfcoll['sdss_id'][dfcoll['sdss_id'].duplicated()]

    for d in dups:
        s = subjects.find({'metadata.sdss_id':d})
        dfs = pd.DataFrame(list(s))
        for z in dfs['zooniverse_id']:
            zc = pd.DataFrame(list(classifications.find({'subjects.zooniverse_id':z})))
            firstclass = zc['created_at'][zc.index[0]]
            lastclass = zc['created_at'][zc.index[-1]]
            td = (lastclass - firstclass)
            print '%s (%s) was created over a period of %02i days (%s to %s))' % (d,z,td.days,firstclass,lastclass)

        print '---------------------------'

    return None



