"""Task-independent conditional tension selection; no strength/activation cap.

Hard constraints: moment balance, imported passive floor, active support and
optional linear reaction-direction stability. A strictly convex temporal
regularizer selects ONE reproducible history, not neural/metabolic optimality.
"""
import numpy as np
import scipy.sparse as sp
import osqp
import quadprog
from scipy.linalg import qr


def scale_parameters(names,reference,gain,passive,profile,groups):
    """Reference-force/PCSA sensitivity only; geometry and body inertia fixed.

    Exact element entries replace the group's multiplier. Overlapping selected
    groups are rejected so one element never receives accidental double scaling.
    """
    names=list(names);scale=np.ones(len(names));assigned=set()
    for group,value in profile.get('group_reference_force_scales',{}).items():
        if group not in groups:raise ValueError('Unknown muscle group: '+group)
        indices=[i for i,name in enumerate(names) if name.startswith(groups[group])]
        if not indices:raise ValueError('Empty muscle group: '+group)
        if assigned.intersection(indices):raise ValueError('Overlapping group scales')
        if not np.isfinite(value) or value<=0:raise ValueError('Invalid positive force scale')
        scale[indices]=value;assigned.update(indices)
    for name,value in profile.get('element_reference_force_scales',{}).items():
        if name not in names:raise ValueError('Unknown muscle element: '+name)
        if not np.isfinite(value) or value<=0:raise ValueError('Invalid positive force scale')
        scale[names.index(name)]=value
    return np.asarray(reference)*scale,np.asarray(gain)*scale,np.asarray(passive)*scale,scale


def reaction_cone(coefficients,offsets,axes,half_angle_deg=20.,faces=16):
    """JRF = offset - coefficients @ total_tension, GH approximation only."""
    if not 0<half_angle_deg<90 or faces<4:raise ValueError('Invalid cone')
    rows=[];bounds=[];apothem=np.tan(np.deg2rad(half_angle_deg))*np.cos(np.pi/faces)
    for S,offset,axis in zip(coefficients,offsets,axes):
        axis=np.asarray(axis);axis=axis/np.linalg.norm(axis);ref=np.array([0.,0,1.])
        if abs(axis@ref)>.9:ref=np.array([0.,1.,0])
        u=np.cross(axis,ref);u/=np.linalg.norm(u);v=np.cross(axis,u)
        directions=[-axis]+[np.cos(phi)*u+np.sin(phi)*v-apothem*axis for phi in np.linspace(0,2*np.pi,faces,endpoint=False)]
        for direction in directions:rows.append(-direction@S);bounds.append(-direction@offset)
    return np.array(rows),np.array(bounds)


def solve_causal_history(times,R,target,gain,passive,reference,stability=None,temporal_weight=.01,level_weight=1.):
    """Streaming reference: magnitude plus change from the previous valid force.

    Every sample satisfies the same physical constraints. This causal numerical
    selection rule is distinct from the batch whole-history minimizer.
    """
    times=np.asarray(times);R=np.asarray(R);target=np.asarray(target);gain=np.asarray(gain);passive=np.asarray(passive);reference=np.asarray(reference);nt,nq,nm=R.shape;forces=[];maximum_error=0.;maximum_stability=0.;previous=None
    if np.any(np.diff(times)<=0) or np.any(reference<=0):raise ValueError('Invalid causal input')
    for i in range(nt):
        active=np.where(gain[i]>=1e-8)[0];n=len(active);floor=passive[i];matrix=R[i][:,active]*reference[active];rhs=target[i]-R[i]@floor;scale=np.maximum(np.linalg.norm(matrix,axis=1),1.);A=matrix/scale[:,None];value=rhs/scale
        _,tri,piv=qr(A.T,pivoting=True,mode='economic');rank=int(np.sum(abs(np.diag(tri))>1e-10));keep=piv[:rank]
        dt=float(times[i]-times[i-1]) if i else 1.;magnitude=level_weight*dt;continuity=temporal_weight/dt if i else 0.;h=magnitude+continuity;linear=2*h*(floor/reference)[active]
        if previous is not None:linear-=2*continuity*(previous/reference)[active]
        rows=[A[keep],np.eye(n)];bounds=[value[keep],np.zeros(n)]
        if stability is not None:
            C,b=stability[i];S=C[:,active]*reference[active];s=np.maximum(np.linalg.norm(S,axis=1),1.);rows.append(-S/s[:,None]);bounds.append(-(b-C@floor)/s)
        constraints=np.vstack(rows);limits=np.concatenate(bounds)
        try:x=quadprog.solve_qp(np.eye(n)/np.sqrt(2*h),-linear,constraints.T,limits,meq=rank,factorized=True)[0]
        except ValueError as error:raise RuntimeError(('Causal tension frame',i,float(times[i]),str(error))) from error
        F=floor.copy();F[active]+=reference[active]*np.maximum(x,0.);error=float(abs(R[i]@F-target[i]).max());violation=max(0.,float((stability[i][0]@F-stability[i][1]).max())) if stability is not None else 0.
        if max(error,violation)>.01:raise RuntimeError(('Causal physical residual',i,error,violation))
        maximum_error=max(maximum_error,error);maximum_stability=max(maximum_stability,violation);forces.append(F);previous=F
    F=np.array(forces);active=np.maximum(F-passive,0);effort=np.divide(active,gain,out=np.zeros_like(active),where=gain>=1e-8)
    return {'force_N':F,'active_force_N':active,'passive_force_N':passive,'unbounded_effort_coefficient':effort,'checks':{'status':'passed','selection_mode':'causal magnitude-and-continuity reference','moment_balance_max_error':maximum_error,'stability_max_violation_N':maximum_stability,'strength_ceiling_enforced':False,'neural_activation_claim':False,'metabolic_energy_claim':False,'temporal_weight':temporal_weight,'level_weight':level_weight,'maximum_effort_coefficient':float(effort.max())}}


def solve_history(times,R,target,gain,passive,reference,stability=None,temporal_weight=.01,level_weight=1.,initial_force=None,time_limit_s=None):
    """Minimize λ∫||d(F/Fref)/dt||²dt + ε∫||F/Fref||²dt.

    The magnitude term fixes the constant nullspace. Neither term is claimed
    to be metabolic energy or measured effort. FLV gain maps the output to an
    unbounded effort coefficient; zero gain forbids active force in that element.
    """
    times=np.asarray(times,float);R=np.asarray(R,float);target=np.asarray(target,float);gain=np.asarray(gain,float);passive=np.asarray(passive,float);reference=np.asarray(reference,float)
    nt,nq,nm=R.shape
    if nt!=len(times) or target.shape!=(nt,nq) or gain.shape!=(nt,nm) or passive.shape!=(nt,nm) or reference.shape!=(nm,):raise ValueError('Inconsistent tension input shapes')
    if not all(np.isfinite(x).all() for x in [times,R,target,gain,passive,reference]) or np.any(reference<=0) or np.any(gain<0) or np.any(passive<0):raise ValueError('Nonfinite or invalid tension input')
    if level_weight<=0 or temporal_weight<0:raise ValueError('Invalid regularization')
    dt=np.diff(times)
    if np.any(dt<=0):raise ValueError('Times must increase')
    quad=np.ones(nt) if nt==1 else np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2]
    timeP=sp.diags(level_weight*quad,format='csc')
    if nt>1:
        D=sp.diags([-np.ones(nt-1),np.ones(nt-1)],[0,1],shape=(nt-1,nt),format='csc')
        timeP+=temporal_weight*D.T@sp.diags(1/dt)@D
    P=2*sp.kron(timeP,sp.eye(nm),format='csc')
    Aeq=sp.block_diag([sp.csc_matrix(r*reference) for r in R],format='csc');eqscale=np.maximum(np.sqrt(np.array(Aeq.power(2).sum(1)).ravel()),1.)
    rows=[sp.diags(1/eqscale)@Aeq];lo=[target.ravel()/eqscale];hi=[lo[0].copy()]
    lower=(passive/reference).ravel();upper=np.full(nt*nm,np.inf);inactive=(gain<1e-8).ravel();upper[inactive]=lower[inactive]
    rows.append(sp.eye(nt*nm,format='csc'));lo.append(lower);hi.append(upper)
    if stability is not None:
        blocks=[sp.csc_matrix(C*reference) for C,b in stability];Ast=sp.block_diag(blocks,format='csc');rhs=np.concatenate([b for C,b in stability]);s=np.maximum(np.sqrt(np.array(Ast.power(2).sum(1)).ravel()),1.)
        rows.append(sp.diags(1/s)@Ast);lo.append(np.full(len(rhs),-np.inf));hi.append(rhs/s)
    A=sp.vstack(rows,format='csc');qp=osqp.OSQP();settings={'time_limit':time_limit_s} if time_limit_s is not None else {};qp.setup(P=P,q=np.zeros(nt*nm),A=A,l=np.concatenate(lo),u=np.concatenate(hi),verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True,**settings)
    if initial_force is not None:
        seed=np.asarray(initial_force,float)
        if seed.shape!=(nt,nm) or not np.isfinite(seed).all():raise ValueError('Invalid warm-start tension shape')
        seed=np.maximum(seed/reference,lower.reshape(nt,nm));seed[gain<1e-8]=lower.reshape(nt,nm)[gain<1e-8];qp.warm_start(x=seed.ravel())
    result=qp.solve(raise_error=False)
    if not result.info.status.startswith('solved'):raise RuntimeError('Tension reference QP: '+result.info.status)
    y=result.x.reshape(nt,nm);F=y*reference;active=F-passive
    moment_error=float(abs(np.einsum('tqn,tn->tq',R,F)-target).max());floor_error=float(np.maximum(-active,0).max());inactive_error=float(abs(active[gain<1e-8]).max()) if inactive.any() else 0.
    stability_error=max((float((C@f-b).max()) for (C,b),f in zip(stability,F)),default=0.) if stability is not None else 0.
    if max(moment_error,floor_error,inactive_error,stability_error)>.01:raise RuntimeError(('Tension physical-unit residual',moment_error,floor_error,inactive_error,stability_error))
    # Remove only solver roundoff, then recheck physical balance.
    active=np.maximum(active,0);active[gain<1e-8]=0;F=passive+active
    final_error=float(abs(np.einsum('tqn,tn->tq',R,F)-target).max())
    final_stability=max((float((C@f-b).max()) for (C,b),f in zip(stability,F)),default=0.) if stability is not None else 0.
    if max(final_error,final_stability)>.01:raise RuntimeError('Roundoff cleanup exceeded tolerance')
    effort=np.divide(active,gain,out=np.zeros_like(active),where=gain>=1e-8)
    change=float(np.sum(np.diff(F/reference,axis=0)**2/dt[:,None])) if nt>1 else 0.
    return {'force_N':F,'active_force_N':active,'passive_force_N':passive,'unbounded_effort_coefficient':effort,'checks':{'status':'passed','moment_balance_max_error':final_error,'stability_max_violation_N':max(0.,final_stability),'passive_floor_max_violation_N':floor_error,'zero_gain_active_roundoff_N':inactive_error,'maximum_effort_coefficient':float(effort.max()),'strength_ceiling_enforced':False,'neural_activation_claim':False,'metabolic_energy_claim':False,'temporal_weight':temporal_weight,'level_weight':level_weight,'normalized_force_change_integral':change,'interpretation':'One conditional tension history selected by magnitude and continuity regularization; not a prediction of minimum physiological effort or energy.'}}
