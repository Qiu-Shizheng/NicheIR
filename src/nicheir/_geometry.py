"""Visible-only multiscale features for a known acquisition boundary.

Angles are relative to the supplied outward cut normal. The complete-reference
hull is deliberately absent from observable_features' interface. Crop coverage
is exact; the observed-cell convex hull is only a measured support proxy.
"""
from functools import lru_cache
import numpy as np
from scipy.spatial import ConvexHull,QhullError,cKDTree

RINGS=np.asarray([0.,.5,1.,2.,3.])
SECTORS=16
SLOTS=32
INPUT_KEYS=('polar_counts','crop_coverage','visible_hull_support','local_observed_counts','query_features','valid')

def canonical_coordinates(xy,theta):
    """Shared absolute coordinates; exact axes and add-before-scale diagonals."""
    xy=np.asarray(xy,dtype='f8')
    k=int(np.rint(theta/(np.pi/4)))
    if theta==k*(np.pi/4):
        nx,ny=[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)][k%8]
        scale=np.sqrt(.5) if k%2 else 1.
        return np.stack([(nx*xy[:,0]+ny*xy[:,1])*scale,(-ny*xy[:,0]+nx*xy[:,1])*scale],1)
    c,s=np.cos(theta),np.sin(theta)
    return np.stack([xy[:,0]*c+xy[:,1]*s,-xy[:,0]*s+xy[:,1]*c],1)

def halfplane_projection(xy,theta):
    return canonical_coordinates(xy,theta)[:,0]

def polar_sectors(canonical):
    """Half-open 16 sectors using signs and cross-products, never atan2.

    Tiny positive/negative x near the y axis retains its correct halfplane.
    The exact axis rays start sectors 0,4,8,12. Extended precision comparisons
    against tan(pi/8) avoid angular roundoff at the remaining sector edges.
    """
    x,y=np.asarray(canonical,dtype='f8').T
    quadrant=np.select([(x>0)&(y>=0),(x<=0)&(y>0),(x<0)&(y<=0)],[0,1,2],default=3)
    a=np.select([quadrant==0,quadrant==1,quadrant==2],[x,y,-x],default=-y).astype(np.longdouble)
    b=np.select([quadrant==0,quadrant==1,quadrant==2],[y,-x,-y],default=x).astype(np.longdouble)
    t=np.sqrt(np.longdouble(2))-1
    local=(b>=t*a).astype(int)+(b>=a).astype(int)+(t*b>=a).astype(int)
    result=4*quadrant+local;result[(x==0)&(y==0)]=0
    return result

def encoded_crop_coverage(distance_over_radius):
    """Float32 area with geometric full/partial state preserved at 1."""
    d=np.asarray(distance_over_radius,dtype='f8');result=annular_crop_coverage(d).astype('f4')
    a=np.sqrt(np.longdouble(.5));c=np.sqrt((1+a)/2);s=np.sqrt((1-a)/2)
    maximum_cosine=np.asarray([1,c,a,s,0,0,0,0,0,0,0,0,s,a,c,1])
    # Outward bound certifies a bin wholly visible despite cosine/product
    # rounding. Whole disks and the left halfplane are exact special cases.
    xmax=np.nextafter((RINGS[1:,None].astype(np.longdouble)*maximum_cosine[None,:]).astype('f8'),np.inf)
    structural_full=(d[...,None,None]>=xmax)|(d[...,None,None]>=RINGS[1:,None])|(maximum_cosine[None,:]==0)
    result[structural_full]=1.
    result[(~structural_full)&(result==1)]=np.nextafter(np.float32(1),np.float32(0))
    return result

def disk_sector_coverage(distance,sectors=SECTORS):
    """Exact area fractions of x<=distance in unit-disk angular sectors."""
    d=np.asarray(distance,dtype='f8')
    assert np.isfinite(d).all() and (d>=0).all()
    shape=d.shape;d=np.minimum(d.ravel(),1.)
    alpha=np.arccos(d);width=2*np.pi/sectors
    result=np.ones((len(d),sectors),dtype='f8')
    for sector in range(sectors):
        left=sector*width;right=(sector+1)*width
        blocked=np.zeros(len(d));radial=np.zeros(len(d))
        for period in [-1,0,1]:
            a=np.maximum(left,-alpha+2*np.pi*period)
            b=np.minimum(right,alpha+2*np.pi*period)
            use=b>a
            if not use.any():continue
            lo=a[use]-2*np.pi*period;hi=b[use]-2*np.pi*period;dd=d[use]
            blocked[use]+=hi-lo
            value=np.zeros(len(dd));positive=dd>0
            value[positive]=(dd[positive]/np.maximum(np.cos(lo[positive]),dd[positive]))*(dd[positive]/np.maximum(np.cos(hi[positive]),dd[positive]))*np.sin((hi-lo)[positive])
            radial[use]+=value
        result[:,sector]=(width-blocked+radial)/width
    result[d>=1]=1.
    return result.clip(0,1).reshape(shape+(sectors,))

def annular_crop_coverage(distance_over_radius):
    d=np.asarray(distance_over_radius,dtype='f8')
    cumulative=[np.zeros(d.shape+(SECTORS,))]
    for radius in RINGS[1:]:
        cumulative.append(radius*radius*disk_sector_coverage(d/radius))
    return np.stack([(cumulative[i+1]-cumulative[i])/(RINGS[i+1]**2-RINGS[i]**2) for i in range(4)],axis=-2).clip(0,1)

@lru_cache(None)
def angular_rule(order):
    x,w=np.polynomial.legendre.leggauss(order)
    angles=(np.arange(SECTORS)[:,None]+(x[None,:]+1)/2)*(2*np.pi/SECTORS)
    return np.stack([np.cos(angles.ravel()),np.sin(angles.ravel())],1),w/2

def observed_hull_support(equations,queries,order=32):
    """Sector-annulus fraction inside the visible-point hull via Gauss angles.

    Query coordinates and hull equations use the same canonical/R coordinate
    frame. Radial intersection is exact at each angle; only angular integration
    is numerical. The hull is an observable proxy, not a tissue mask.
    """
    directions,weights=angular_rule(order)
    clearance=-(queries@equations[:,:2].T+equations[:,2])
    assert clearance.min()>=-1e-7
    clearance=clearance.clip(0)
    dot=equations[:,:2]@directions.T
    cap=np.full((len(queries),len(equations),len(directions)),np.inf)
    np.divide(clearance[:,:,None],dot[None,:,:],out=cap,where=dot[None,:,:]>1e-14)
    radius=cap.min(1)
    fractions=[]
    for lo,hi in zip(RINGS[:-1],RINGS[1:]):
        radial=((np.minimum(radius,hi)**2-lo*lo)/(hi*hi-lo*lo)).clip(0,1)
        fractions.append((radial.reshape(len(queries),SECTORS,order)*weights).sum(2))
    return np.stack(fractions,1).clip(0,1)

def analytic_hull_support(vertices,queries):
    """Exact annular-sector area within a convex observed-point polygon.

    Vertices are counterclockwise, queries are in/on the polygon, all in R
    units. Vertex rays fix the intersected edge on each angular interval;
    edge/circle intersections fix whether radial clipping is active. On a
    partial interval, integral (h/cos(theta-phi))**2 dtheta is evaluated as
    (h/cos(a-phi))*(h/cos(b-phi))*sin(b-a), avoiding tangent cancellation.
    Boundary anchors need no sampling or special eligibility rule.
    """
    vertices=np.asarray(vertices,dtype='f8');queries=np.asarray(queries,dtype='f8')
    edge=np.roll(vertices,-1,axis=0)-vertices
    lengths=np.linalg.norm(edge,axis=1);assert (lengths>0).all()
    normals=np.stack([edge[:,1],-edge[:,0]],axis=1)/lengths[:,None]
    phi=np.arctan2(normals[:,1],normals[:,0])
    width=2*np.pi/SECTORS;period=2*np.pi
    result=np.zeros((len(queries),4,SECTORS),dtype='f8')
    for query_index,query in enumerate(queries):
        clearance=((vertices-query)*normals).sum(1)
        assert clearance.min()>=-1e-7
        clearance=clearance.clip(0)
        offset=vertices-query
        breaks=[np.arange(SECTORS+1)*width,np.mod(np.arctan2(offset[:,1],offset[:,0]),period)]
        for radius in RINGS[1:]:
            use=clearance<=radius
            angle=np.arccos(np.minimum(clearance[use]/radius,1.))
            breaks.extend([np.mod(phi[use]+angle,period),np.mod(phi[use]-angle,period)])
        bounds=np.unique(np.concatenate(breaks));a=bounds[:-1];b=bounds[1:]
        middle=(a+b)/2;delta=b-a
        dots=normals@np.stack([np.cos(middle),np.sin(middle)])
        cap=np.full(dots.shape,np.inf)
        np.divide(clearance[:,None],dots,out=cap,where=dots>0)
        active=np.argmin(cap,axis=0);radial=cap[active,np.arange(len(middle))]
        assert np.isfinite(radial).all()
        sectors=np.minimum((middle/width).astype(int),SECTORS-1)
        h=clearance[active];active_phi=phi[active]
        for ring,(lo,hi) in enumerate(zip(RINGS[:-1],RINGS[1:])):
            scale=hi*hi-lo*lo
            area=np.where(radial>=hi,scale*delta,0.)
            partial=(radial>lo)&(radial<hi)
            if partial.any():
                hp=h[partial];pa=a[partial]-active_phi[partial];pb=b[partial]-active_phi[partial]
                # Circle-intersection split guarantees endpoint distances <=hi;
                # max only contains endpoint roundoff at the exact transition.
                radius_a=hp/np.maximum(np.cos(pa),hp/hi)
                radius_b=hp/np.maximum(np.cos(pb),hp/hi)
                area[partial]=radius_a*radius_b*np.sin(delta[partial])-lo*lo*delta[partial]
            area=np.clip(area,0,scale*delta)
            result[query_index,ring]=np.bincount(sectors,weights=area,minlength=SECTORS)/(scale*width)
    assert np.isfinite(result).all()
    return result.clip(0,1)

def polar_count(offsets,types,theta,max_radius=3.,canonical_offsets=None):
    """Offsets already in R units; center object must have been removed."""
    canonical=canonical_coordinates(offsets,theta) if canonical_offsets is None else canonical_offsets
    distance=np.linalg.norm(offsets,axis=1)
    known=(types>=0)&(distance<=max_radius)
    distance=distance[known];labels=types[known]
    sectors=polar_sectors(canonical[known])
    rings=np.searchsorted(RINGS[1:],distance,side='left')
    return np.bincount((rings*SECTORS+sectors)*4+labels,minlength=4*SECTORS*4).reshape(4,SECTORS,4).astype('f4')

def observable_features(visible_xy,visible_types,visible_ids,query_ids,theta,cut,radius,full=False):
    """Only actual visible cells and a known acquisition boundary are accepted."""
    xy=np.asarray(visible_xy,dtype='f8');types=np.asarray(visible_types,dtype=int)
    ids=np.asarray(visible_ids).astype(str);query_ids=np.asarray(query_ids).astype(str)
    assert len(ids)==len(set(ids)) and len(query_ids)<=SLOTS
    assert np.isfinite(xy).all() and np.isin(types,[-1,0,1,2,3]).all()
    normal=np.asarray([np.cos(theta),np.sin(theta)])
    absolute_canonical=canonical_coordinates(xy,theta)
    projection=absolute_canonical[:,0]
    if not full:assert (projection<=cut+1e-9).all(),'Hidden cells passed into observable features'
    lookup={identifier:i for i,identifier in enumerate(ids)}
    qi=np.asarray([lookup[k] for k in query_ids]);assert (types[qi]==3).all()
    canonical=(absolute_canonical-absolute_canonical.mean(0))/radius
    qxy=canonical[qi]
    try:hull=ConvexHull(canonical)
    except QhullError:raise ValueError('Visible support hull is degenerate')
    support=analytic_hull_support(canonical[hull.vertices],qxy)
    x={key:np.zeros(shape,dtype='f4') for key,shape in [
        ('polar_counts',(SLOTS,4,SECTORS,4)),('crop_coverage',(SLOTS,4,SECTORS)),
        ('visible_hull_support',(SLOTS,4,SECTORS)),('local_observed_counts',(SLOTS,4)),
        ('query_features',(SLOTS,7)),('valid',(SLOTS,))]}
    count=len(qi);distance=np.full(count,3.) if full else np.maximum((cut-projection[qi])/radius,0)
    x['crop_coverage'][:count]=encoded_crop_coverage(distance)
    x['visible_hull_support'][:count]=support
    assert np.max(support-x['crop_coverage'][:count])<=2e-6,'Observed hull extends outside known crop'
    x['query_features'][:count,:3]=np.column_stack([distance.clip(0,3),qxy])
    x['query_features'][:count,6]=1.
    x['valid'][:count]=1.
    tree=cKDTree(xy)
    for slot,index in enumerate(qi):
        neighbors=np.asarray(tree.query_ball_point(xy[index],3*radius),dtype=int)
        neighbors=neighbors[neighbors!=index]
        x['polar_counts'][slot]=polar_count((xy[neighbors]-xy[index])/radius,types[neighbors],theta,canonical_offsets=(absolute_canonical[neighbors]-absolute_canonical[index])/radius)
    x['local_observed_counts']=x['polar_counts'][:,:2].sum((1,2))
    assert set(x)==set(INPUT_KEYS)
    return x,dict(support_method='analytic_convex_polygon_annulus',support_crop_excess=float(np.maximum(support-x['crop_coverage'][:count],0).max()),visible_cells=len(xy),visible_known_cells=int((types>=0).sum()))
