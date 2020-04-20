#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:17:21 2020

@author: yumin
"""
#%% BCSD
folders = {'ppt':{'BCSD':{2:'',4:'',8:''},
                 'DeepSD':{2:'2020-01-08_15.36.51.715443',4:'2020-01-08_15.46.06.714138',8:'2020-01-08_16.06.29.813876'},
                 'ESPCN':{2:'2020-01-09_10.31.57.916833',4:'2020-01-09_10.11.18.969498',8:'2020-01-08_18.18.38.602588'},
                 'REDNet':{2:'2020-01-09_10.58.54.097537',4:'2020-01-09_11.51.35.785891',8:'2020-01-09_12.22.56.720490'},
                 'YNet':{2:'2020-01-08_14.44.21.689965_debug',4:'2020-01-08_14.00.55.945108_debug',8:'2020-01-08_11.03.16.421380_debug'}
                 },

          'tmax':{'BCSD':{2:'',4:'',8:''},
                 'DeepSD':{2:'2019-12-12_14.58.50.069989',4:'2019-12-12_15.34.38.945006',8:'2019-12-12_15.47.44.177582'},
                 'ESPCN':{2:'2019-12-19_16.53.43.975465',4:'2019-12-19_19.19.58.832526',8:'2019-12-19_19.31.37.850150'},
                 'REDNet':{2:'2019-12-12_16.57.49.371496',4:'2019-12-12_17.08.14.793995',8:'2019-12-12_17.30.31.506155'},
                 'YNet':{2:'2019-12-16_14.40.18.526317_debug',4:'2019-12-16_15.19.15.450559_debug',8:'2019-12-16_11.04.30.215301_debug'}
                  },
          
          'tmin':{'BCSD':{2:'',4:'',8:''},
                 'DeepSD':{2:'2019-12-17_22.18.06.316037',4:'2019-12-17_22.06.53.135085',8:'2019-12-17_21.33.30.402814'},
                 'ESPCN':{2:'2019-12-19_16.19.01.765814',4:'2019-12-19_16.04.03.388721',8:'2019-12-19_15.53.40.591203'},
                 'REDNet':{2:'2019-12-18_22.35.35.394491',4:'2019-12-18_12.34.27.547029',8:'2019-12-18_11.02.12.469294'},
                 'YNet':{2:'2019-12-19_10.25.40.824423_debug',4:'2019-12-19_09.22.51.164536_debug',8:'2019-12-18_22.50.07.662488_debug'}
                  }
          }
prednames = {'ppt':{'BCSD':{2:'gcms_bcsd_pred_MSE1.5543249278666207',4:'gcms_bcsd_pred_MSE1.688033953263116',8:'gcms_bcsd_pred_MSE1.7627197220364208'},
                   'DeepSD':{2:'pred_results_MSE1.5226051807403564',4:'pred_results_MSE1.681918740272522',8:'pred_results_MSE1.7008076906204224'},
                   'ESPCN':{2:'pred_results_MSE1.9864422116014693',4:'pred_results_MSE2.2170647746986814',8:'pred_results_MSE2.4984338548448353'},
                   'REDNet':{2:'pred_results_MSE1.3450590305858188',4:'pred_results_MSE1.4671132332748837',8:'pred_results_MSE1.5432003852393892'},
                   'YNet':{2:'pred_results_MSE1.2912344543470278',4:'pred_results_MSE1.3957129584418402',8:'pred_results_MSE1.4542109436459012'}
                   },
          
            'tmax':{'BCSD':{2:'gcms_bcsd_pred_MSE7.338524662379835',4:'gcms_bcsd_pred_MSE7.2560945786866045',8:'gcms_bcsd_pred_MSE7.288684181494683'},
                    'DeepSD':{2:'pred_results',4:'pred_results',8:'pred_results'},
                    'ESPCN':{2:'pred_results_MSE21.353439993328518',4:'pred_results_MSE24.279992792341446',8:'pred_results_MSE29.211670928531223'},
                   'REDNet':{2:'pred_results',4:'pred_results',8:'pred_results'},
                   'YNet':{2:'pred_results_MSE2.3648972941769495',4:'pred_results_MSE2.610652203361193',8:'pred_results_MSE2.5781057857804828'}
                    },
            
            'tmin':{'BCSD':{2:'gcms_bcsd_pred_MSE7.188308651740791',4:'gcms_bcsd_pred_MSE6.9078619691877865',8:'gcms_bcsd_pred_MSE6.689920848053531'},
                    'DeepSD':{2:'pred_results_MSE3.5547709465026855',4:'pred_results_MSE3.460289716720581',8:'pred_results_MSE3.8001818656921387'},
                    'ESPCN':{2:'pred_results_MSE11.34368884563446',4:'pred_results_MSE13.027198592821756',8:'pred_results_MSE13.988994240760803'},
                   'REDNet':{2:'pred_results_MSE1.80604773428705',4:'pred_results_MSE2.4965724680158825',8:'pred_results_MSE2.693539967139562'},
                   'YNet':{2:'pred_results_MSE2.0703872276677027',4:'pred_results_MSE2.0897242344088025',8:'pred_results_MSE2.0874277618196277'}
                    }
            }



# =============================================================================
#     predpath = '../results/Climate/PRISM_GCM/YNet30/{}/scale{}/{}/'.format(variable,scale,folder)
#     predpath = '../results/Climate/PRISM_GCM/REDNet30/{}/scale{}/{}/'.format(variable,scale,folder)
#     predpath = '../results/Climate/PRISM_GCM/ESPCN/{}/scale{}/{}/'.format(variable,scale,folder)
#     predpath = '../results/Climate/PRISM_GCM/DeepSD/{}/train_together/{}by{}/{}/'.format(variable,resolution,resolution,folder)
#     predpath = '../results/Climate/PRISM_GCM/BCSD/{}/{}by{}/'.format(variable,resolution,resolution)
#     
#     
#     preds = np.load(predpath+predname+'.npy')
#     savepath = '../results/Climate/PRISM_GCM/Figures/abs_diff/' # None
#     #names = {8:'pred_results_MSE1.5432003852393892'}
#     #folders = {8:'2020-01-09_12.22.56.720490'}
#     #%% USA mask
#     datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
#     maskdatapath = datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution)
# =============================================================================





















