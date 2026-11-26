"""Explicit anatomical-name to model actuator-group mapping; no invented loads."""
def map_atlas(name,actuators):
 n=name.lower();side='l' if 'left' in n else 'r' if 'right' in n else None
 # The torso model's MF actuators represent lumbar multifidus, not cervical or
 # thoracic surfaces newly added for anatomical completeness.
 if 'multifidus' in n and any(x in n for x in ['cervicis','thoracis']):return []
 def arm(prefixes):
  return [i for i,a in enumerate(actuators) if any(a==(p+'_l' if side=='l' else p) for p in prefixes)]
 def body(prefixes):
  return [i for i,a in enumerate(actuators) if (side is None or a.endswith('_'+side)) and any(a.startswith(p) for p in prefixes)]
 for phrase,prefix in [('trapezius','Trapezius'),('serratus anterior','Serratus'),('rhomboid','Rhomboideus'),('levator scapulae','Levator'),('pectoralis minor','PectoralisMinor')]:
  if phrase in n:return body([prefix])
 if 'deltoid' in n:return arm(['DELT1' if 'clavicular' in n else 'DELT2' if 'acromial' in n else 'DELT3' if 'spinal' in n else 'DELT1','DELT2'] if not any(k in n for k in ['clavicular','acromial','spinal']) else ['DELT1' if 'clavicular' in n else 'DELT2' if 'acromial' in n else 'DELT3'])
 if 'pectoralis major' in n:return arm(['PECM1' if 'clavicular' in n else 'PECM3' if 'abdominal' in n else 'PECM2'])
 if 'latissimus' in n:return arm(['LAT1','LAT2','LAT3'])
 if 'triceps brachii' in n:return arm(['TRIlong' if 'long head' in n else 'TRIlat' if 'lateral head' in n else 'TRImed' if 'medial head' in n else 'TRIlong','TRIlat','TRImed'] if not any(k in n for k in ['long head','lateral head','medial head']) else ['TRIlong' if 'long head' in n else 'TRIlat' if 'lateral head' in n else 'TRImed'])
 if 'biceps brachii' in n:return arm(['BICshort'] if 'short head' in n else ['BIClong'] if 'long head' in n else ['BIClong','BICshort'])
 for phrase,prefixes in [('supraspinatus',['SUPSP']),('infraspinatus',['INFSP']),('subscapularis',['SUBSC']),('teres major',['TMAJ']),('teres minor',['TMIN']),('coracobrachialis',['CORB']),('brachioradialis',['BRD']),('brachialis',['BRA']),('anconeus',['ANC']),('supinator',['SUP']),('pronator teres',['PT']),('pronator quadratus',['PQ']),('flexor carpi radialis',['FCR']),('flexor carpi ulnaris',['FCU']),('palmaris longus',['PL']),('extensor carpi radialis longus',['ECRL']),('extensor carpi radialis brevis',['ECRB']),('extensor carpi ulnaris',['ECU']),('flexor digitorum superficialis',['FDS2','FDS3','FDS4','FDS5']),('flexor digitorum profundus',['FDP2','FDP3','FDP4','FDP5']),('extensor digitorum',['EDC2','EDC3','EDC4','EDC5']),('extensor indicis',['EIP']),('extensor digiti minimi',['EDM']),('flexor pollicis longus',['FPL']),('extensor pollicis longus',['EPL']),('extensor pollicis brevis',['EPB']),('abductor pollicis longus',['APL'])]:
  if phrase in n:return arm(prefixes)
 for phrase,prefixes in [('rectus abdominis',['rect_abd']),('external oblique',['EO']),('internal oblique',['IO']),('iliocostalis',['IL_']),('longissimus',['LTp']),('multifidus',['MF_']),('quadratus lumborum',['QL_']),('psoas',['Ps_','psoas_']),('iliacus',['iliacus_']),('gluteus maximus',['glmax']),('gluteus medius',['glmed']),('gluteus minimus',['glmin']),('rectus femoris',['recfem']),('vastus lateralis',['vaslat']),('vastus medialis',['vasmed']),('vastus intermedius',['vasint']),('biceps femoris',['bfsh'] if 'short head' in n else ['bflh'] if 'long head' in n else ['bflh','bfsh']),('semimembranosus',['semimem']),('semitendinosus',['semiten']),('sartorius',['sart_']),('gracilis',['grac_']),('adductor longus',['addlong']),('adductor brevis',['addbrev']),('adductor magnus',['addmag']),('tensor fasciae latae',['tfl_']),('piriformis',['piri_']),('gastrocnemius',['gaslat'] if 'lateral head' in n else ['gasmed'] if 'medial head' in n else ['gaslat','gasmed']),('soleus',['soleus_']),('tibialis anterior',['tibant']),('tibialis posterior',['tibpost']),('fibularis longus',['perlong']),('peroneus longus',['perlong']),('fibularis brevis',['perbrev']),('peroneus brevis',['perbrev']),('extensor digitorum longus',['edl_']),('extensor hallucis longus',['ehl_']),('flexor digitorum longus',['fdl_']),('flexor hallucis longus',['fhl_'])]:
  if phrase in n:return body(prefixes)
 return []
