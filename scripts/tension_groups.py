"""Nonoverlapping display groups covering every native muscle actuator."""
MASK_STEP=4  # 50 group/side IDs fit in the existing 8-bit numerical mask.
REGISTRY=[
 ('deltoid','Deltoid',('DELT',),'DELT2'),('pectoralis','Pectoralis',('PECM',),'PECM2'),
 ('triceps','Triceps',('TRI',),'TRIlong'),('biceps','Biceps',('BIC',),'BIClong'),
 ('rotator_cuff','Rotator cuff',('SUPSP','INFSP','SUBSC','TMIN'),'INFSP'),
 ('latissimus','Latissimus',('LAT',),'LAT2'),
 ('abdominals','Abdominal wall',('rect_abd','EO','IO'),'rect_abd'),
 ('spinal_extensors','Spinal extensors',('IL_','LTp','MF_'),'IL_L3'),
 ('gluteals','Gluteals',('glmax','glmed','glmin'),'glmax2'),
 ('quadriceps','Quadriceps',('vas','recfem'),'vaslat'),
 ('hamstrings','Hamstrings',('bflh','bfsh','semimem','semiten'),'semimem'),
 ('gastrocnemius','Gastrocnemius',('gas',),'gasmed'),
 ('soleus','Soleus',('soleus',),'soleus'),
 ('hip_flexors','Hip flexors',('psoas','iliacus','Ps_'),'iliacus'),
 ('adductors','Adductors',('addlong','addbrev','addmag','grac'),'addlong'),
 ('forearm','Forearm / extrinsic hand',('FCR','FCU','ECRL','ECRB','ECU','PT','PQ','FDS','FDP','EDC','PL','EDM','EIP','EPL','EPB','FPL','APL'),'FCR'),
 ('tibialis_anterior','Tibialis anterior',('tibant',),'tibant'),
 ('tibialis_posterior','Tibialis posterior',('tibpost',),'tibpost'),
 ('fibularis','Fibularis',('perlong','perbrev'),'perlong'),
 ('toe_flexors','Extrinsic toe flexors',('fdl','fhl'),'fhl'),
 ('toe_extensors','Extrinsic toe extensors',('edl','ehl'),'edl'),
 ('quadratus_lumborum','Quadratus lumborum',('QL_',),'QL_post_I.1-L3'),
 ('hip_accessory','Accessory hip muscles',('piri','sart','tfl'),'tfl'),
 ('arm_accessory','Accessory arm muscles',('TMAJ','CORB','ANC','=SUP','BRA','BRD'),'BRA'),
 ('hand_intrinsics','Hand intrinsics',('OP','RI','LU_','UI_'),'OP')]

def matches(name,prefixes):
    base=name[:-2] if name.endswith(('_r','_l')) else name
    return any(base==p[1:] if p.startswith('=') else name.startswith(p) for p in prefixes)

OVERVIEW=[
 ('Gluteals',['gluteals']),('Quadriceps',['quadriceps']),('Hamstrings',['hamstrings']),
 ('Gastrocnemius',['gastrocnemius']),('Soleus',['soleus']),('Tibialis anterior',['tibialis_anterior']),
 ('Ankle stabilizers',['tibialis_posterior','fibularis']),('Toe flexors / extensors',['toe_flexors','toe_extensors']),
 ('Abdominal wall',['abdominals']),('Back / lumbar',['spinal_extensors','quadratus_lumborum']),
 ('Hip support',['hip_flexors','adductors','hip_accessory']),('Shoulder / chest',['deltoid','pectoralis','rotator_cuff']),
 ('Arm / forearm',['triceps','biceps','arm_accessory','forearm','hand_intrinsics']),('Foot intrinsics',[])]
