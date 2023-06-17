import zipfile
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import geopy.distance
from matplotlib import cm
from io import StringIO
import json

from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import numpy as np

from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import numpy as np

from argparse import ArgumentParser

def options():
    parser = ArgumentParser()

    parser.add_argument('line_id')
    parser.add_argument('savefile')

    return parser.parse_args()

def plot_dendrogram(model, plot=False, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    order = dendrogram(linkage_matrix, no_plot = not plot, get_leaves=True, **kwargs)['leaves']
    return linkage_matrix, order


def elbow(data):
    elbowdata = np.array(data)
    if elbowdata.shape[1] != 2:
        elbowdata = elbowdata.T
    elbowdata -= elbowdata[0]
    rotation = np.linalg.inv(np.vstack([elbowdata[-1], [-elbowdata[-1,1], elbowdata[-1,0]]]).T)
    idx = np.argmin((elbowdata @rotation.T)[:,1])
    return idx    


def autocluster(data, return_order=False, return_linkage=False, plot=False, **kwargs):
    ag = AgglomerativeClustering(distance_threshold=0,n_clusters=None,**kwargs)
    ag.fit(data)
    linkage_matrix, order = plot_dendrogram(ag)
    idx = len(linkage_matrix) - elbow(list(enumerate(linkage_matrix[:,2])))
    if plot:
        plt.plot(linkage_matrix[:,2][::-1])
        plt.show()
    print("number of clusters: ", idx)
    ag2 = AgglomerativeClustering(n_clusters=idx, **kwargs)
    ag2.fit(data)
    ret = [ag2]
    if return_order:
        ret.append(order)
    if return_linkage:
        ret.append(linkage_matrix)
    return tuple(ret) if len(ret) > 1 else ret[0]

def cluster_grouper(ag, data):
    cluster_groups = [[] for _ in range(ag.n_clusters_)]
    for i,stop in enumerate(data):
        if i >= len(ag.labels_):
            break
        cluster_groups[ag.labels_[i]].append(stop)
    clusters = []
    for i,gp in enumerate(tqdm(cluster_groups, desc="initial cluster assignment")):
        lat = np.mean([g['lat'] for g in gp])
        lng = np.mean([g['lng'] for g in gp])
        rads = [ np.linalg.norm(ll2euclid(lat, lng) - ll2euclid(g['lat'], g['lng'])) for g in gp]
        lat = np.mean([g['lat'] for g,r in zip(gp, rads)])
        lng = np.mean([g['lng'] for g,r in zip(gp, rads)])
        duration = sum([g['duration'] for g,r in zip(gp, rads)])
        durations = [g['duration'] for g,r in zip(gp, rads)]
        
    #     radius = max([r for g,r in zip(gp, rads)])
        counts = len(set([g['id'] for g,r in zip(gp, rads)]))
        clusters.append( dict(lat=lat, lng=lng, duration=duration, counts=counts, durations=durations))#, radius=radius))
    cluster_groups = [[] for _ in range(ag.n_clusters_)]
    for i,stop in enumerate(tqdm(data, desc='full cluster assignment')):
        if i < len(ag.labels_):
            cluster_groups[ag.labels_[i]].append(stop)    
        else:
            dists = [ np.linalg.norm(ll2euclid(stop['lat'], stop['lng']) - ll2euclid(cl['lat'], cl['lng'])) for cl in clusters ]
            cluster_groups[np.argmin(dists)].append(stop)
    return cluster_groups

def plot_abq(data, first=True, **kwargs):
    if first:
        print('loading image')
        img = Image.open('abqmap2.png')
        plt.figure(figsize=(10,10))
        plt.imshow(img)
    for row in data:
        if 'latitude' in row:
            plot(row['latitude'], row['longitude'], **kwargs)
        else:
            plot(*row, **kwargs)
def plot(lat,lon,c, **kwargs):
    A = np.array([[ 3.30048468e+00, -1.76873312e+03],
       [ 1.44506111e+03, -7.69612341e+00],
       [ 1.54147586e+05,  6.14923771e+04]]).T
    pltx = A[1]@[lat,lon,1]
    plty = A[0]@[lat,lon,1]
    plt.scatter([plty], [pltx], color=c, **kwargs)

# right, need to convert lat/lon to euclidean
origin_x = -106.544256
origin_y = 35.034674
foot_x = 1
d = geopy.distance.geodesic( (origin_y, origin_x), (origin_y, origin_x + foot_x)).feet
while abs(d-1) > 1e-5:
    if d > 1:
        foot_x *= 0.9
    else:
        foot_x *= 1.05
        
    d = geopy.distance.geodesic( (origin_y, origin_x), (origin_y, origin_x + foot_x)).feet
foot_y = 1
d = geopy.distance.geodesic( (origin_y, origin_x), (origin_y+foot_y, origin_x)).feet
while abs(d-1) > 1e-5:
    if d > 1:
        foot_y *= 0.9
    else:
        foot_y *= 1.05
    d = geopy.distance.geodesic( (origin_y, origin_x), (origin_y+foot_y, origin_x)).feet
    

def ll2euclid(ptx, pty):
    pt = np.array([ptx, pty])
    pt -= [origin_y, origin_x]
    pt /= [foot_y, foot_x]
    return pt    

def trip2array(trip, n):
    locations = np.zeros((n,2))
    msg_times = np.array([str2time(t) for t in trip['msg_time']])
    for i,t in zip(range(n), np.linspace(min(msg_times),max(msg_times),n, endpoint=False)):
        x = np.argmax(msg_times > t)
        locations[i] = ((trip['latitude'].iloc[x] + trip['latitude'].iloc[x-1])/2, (trip['longitude'].iloc[x] + trip['longitude'].iloc[x-1])/2)
    return locations

def str2time(s):
    hour,minute,second = list(map(int,s.split(':')))
    return second + 60*minute + 60*60*hour


with open('stoplocations.json') as file:
    stoplocations = json.load(file)
    
with open('stopnames.json') as file:
    stopnames = json.load(file)    
def uniq(li):
    output = [li[0]]
    for l in li[1:]:
        if l not in output:
            output.append(l)
    return output


def intersection_label(lat, lng):
    stop_distances = [ (id, geopy.distance.geodesic((lat,lng),(x['lat'], x['lng'])).feet) for id,x in stoplocations.items()]
    closest_stop, d = min(stop_distances, key=lambda x: x[1])
    if d > 100:
        return f'{round(lat,5)}, {round(lng,5)}', None
    else:
        return stopnames[closest_stop], closest_stop    
def full_stop_info(series, limit=5):
    locs=[]
    in_stop=False
    duration = 0
    
    for _,s in series.iterrows():
        if s.speed_mph<limit:
            if not in_stop:
                locs.append( {'lat': s.latitude, 'lng': s.longitude, 'duration': 0})
                in_stop = True
                last_time = str2time(s.msg_time)
            else:
                locs[-1]['duration'] += str2time(s.msg_time) - last_time
                last_time = str2time(s.msg_time)
        elif s.speed_mph >= limit:
            in_stop = False
    return locs, str2time(series.msg_time.iloc[-1]) - str2time(series.msg_time.iloc[0])
def filter_trips(arts):
    eb_allstops = {}
    for art in tqdm(arts):
        for stop in set(art.next_stop_id.tolist()):
            eb_allstops[stop] = eb_allstops.get(stop, 0) + 1
    eb_allstops = [ s for s,c in eb_allstops.items() if c > len(arts)/2]
    priority = np.zeros((len(eb_allstops), len(eb_allstops)))

        
    for art in tqdm(arts):
        order = uniq(art.next_stop_id.tolist())
        order = [o for o in order if o in eb_allstops]
        for i,s1 in enumerate(order):
            idx1 = eb_allstops.index(s1)
            for s2 in order[i+1:]:
                idx2 = eb_allstops.index(s2)
    #             assert idx1 != idx2
                priority[idx1,idx2] += 1
    counter = np.zeros(priority.shape)
    for art in tqdm(arts):
        order = uniq(art.next_stop_id.tolist())
        order = [o for o in order if o in eb_allstops]
        for i,s1 in enumerate(order):
            idx1 = eb_allstops.index(s1)
            for s2 in order:
                idx2 = eb_allstops.index(s2)
    #             assert idx1 != idx2
                counter[idx1,idx2] += 1
    priority = priority / counter    

    eb_order = np.argsort(np.linalg.norm(priority, axis=0, ord=1))
    eb_stops = [eb_allstops[i] for i in eb_order ]

    # now I know the stop order
    valid_arts = []
    # include only buses that come within 500 feet of each stop, in order
    stop_euclid_locs = [ ll2euclid(stoplocations[s]['lat'], stoplocations[s]['lng']) for s in eb_stops]
    for art in tqdm(arts, desc='filtering'):
        locs = np.array([ll2euclid(row.latitude, row.longitude) for _,row in art.iterrows()])
        for stop in stop_euclid_locs:
            distances = np.linalg.norm(locs - stop, axis=1)
            stop_idx = np.argmin(distances)
            if distances[stop_idx] > 700:
                break
            locs = locs[stop_idx:]
        else:
            valid_arts.append(art)
    # breakpoint()
    return valid_arts, eb_stops


def stop_analysis(arts, stops):
## I could also just identify the spots where it stops
    art_stops = []
    speed_limit = 0.1
    stupid_trips = []
    stop_euclid_locs = [ ll2euclid(stoplocations[s]['lat'], stoplocations[s]['lng']) for s in stops]
    for idx, art in enumerate(tqdm(arts, desc="finding stops")):
    #     art = df[df.trip_id == artid]
        art = art.iloc[np.argsort(art.msg_time)]
        prev_speed = 100
        lat = 0
        lng = 0
        stop_start = -1
        duration = 0
        for _,row in art.iterrows():
            # eastbound = art.iloc[0].longitude < art.iloc[-1].longitude
            if row.speed_mph < speed_limit and 100 < min(np.linalg.norm(stop_euclid_locs - ll2euclid(row.latitude,row.longitude), axis=1)):
    #             if prev_speed < speed_limit:
                    duration_delta = str2time(row.msg_time) - stop_start
                    duration += duration_delta
                    lat += duration_delta * row.latitude
                    lng += duration_delta * row.longitude
    #                 stop_start = str2time(row.msg_time)
    #                 continue
    #             elif duration > 0:
    #                 stop_start = str2time(row.msg_time)
    #                 stop_location = (lat / duration, lng/duration)
                    
    # #                 prev_speed = 0
                    
    #                 duration = 0
    #                 lat = 0
    #                 lng = 0
            else:
                if duration > 0:
                    art_stops.append(((lat / duration, lng/duration), duration, idx))
                    if duration > 40000:
                        stupid_trips.append((arts, art_stops[-1]))
                    lat = 0
                    lng = 0
                    duration = 0
                    
                    
            stop_start = str2time(row.msg_time)
            prev_speed = row.speed_mph

    art_stop_info = []
    for a in tqdm(art_stops, desc="processing stops"):
        lat = a[0][0]
        lng = a[0][1]
        duration = a[1]
        
        dist_to_station = min([geopy.distance.geodesic((lat,lng), (stoplocations[b]['lat'], stoplocations[b]['lng'])).feet for b in stops])
        art_stop_info.append(dict(lat=lat, lng=lng, duration=duration, dist_to_station=dist_to_station, id=a[2]))

    far_stops = [ a for a in art_stop_info if a['dist_to_station'] > 100 ]
    np.random.shuffle(far_stops)
    far_stops_euclid = [ ll2euclid(d['lat'], d['lng'])  for d in far_stops]

    ag = autocluster(far_stops_euclid[:20000])
    # labels = np.zeros(len(far_stops_euclid))
    cluster_groups = cluster_grouper(ag, far_stops)
    max_radius = 300
    clusters = []
    for i,gp in enumerate(tqdm(cluster_groups, desc='computing clusters')):
        lat = np.mean([g['lat'] for g in gp])
        lng = np.mean([g['lng'] for g in gp])
        rads = [ np.linalg.norm(ll2euclid(lat, lng) - ll2euclid(g['lat'], g['lng'])) for g in gp]
        lat = np.mean([g['lat'] for g,r in zip(gp, rads) if r < max_radius])
        lng = np.mean([g['lng'] for g,r in zip(gp, rads) if r < max_radius])
        duration = sum([g['duration'] for g,r in zip(gp, rads) if r < max_radius])
        durations = [g['duration'] for g,r in zip(gp, rads) if r < max_radius]
        
    #     radius = max([r for g,r in zip(gp, rads) if r < max_radius])
        counts = len(set([g['id'] for g,r in zip(gp, rads) if r < max_radius]))
        clusters.append( dict(lat=lat, lng=lng, duration=duration, counts=counts, durations=durations))#, radius=radius))
    return clusters


def segment_analysis(arts, stops):
    # all pairs
    segment_info = {}
    stop_info = {}
    segment_stop_infos = []
    # distance_to_stops = []
    stop_distance_threshold = 75
    segments_by_stop = {}
    for art in tqdm(arts, desc="dividing into segments"):
        for idx2 in range(1,len(stops)):
            idx1 = idx2-1
            stop1 = stops[idx1]
            stop1_loc = stoplocations[stop1]
            stop1_name = stopnames[stop1]
            stop2 = stops[idx2]
            stop2_name = stopnames[stop2]
            stop2_loc = stoplocations[stop2]
            # segments = []
            # all_segments = []

            # order = uniq(art.next_stop_id.tolist())
            # if stop2 not in order:
            #     continue
            # order = order[order.index(stop2):]
            # if stop1 in order:
            #     continue
            good = True
            d = [geopy.distance.geodesic((stop1_loc['lat'], stop1_loc['lng']), (row.latitude, row.longitude)).feet if row.speed_mph == 0 else 1000 for _,row in art.iterrows() ]
    #         distance_to_stops.append(d)
            if min(d) > stop_distance_threshold:
                # print(f"didn't stop near {stop1_name}")
                # continue
                good = False
            stop1_idx = np.argmin(d)
            d = [geopy.distance.geodesic((stop2_loc['lat'], stop2_loc['lng']), (row.latitude, row.longitude)).feet if row.speed_mph == 0 else 1000 for _,row in art.iterrows() ]
    #         distance_to_stops.append(d)
            if min(d) > stop_distance_threshold:
                # print(f"didn't stop near {stop2_name}")
                # continue
                good = False
            stop2_idx = np.argmin(d)
            if stop1_idx >= stop2_idx:
                # print("stops in wrong order")
                continue
            # start after we begin moving
            stop1_idx += np.argmax(art.speed_mph.iloc[stop1_idx:stop2_idx] > 0)
            # stop when we stop moving
            stop2_idx -= np.argmax(art.speed_mph.iloc[stop2_idx:stop1_idx:-1] > 0)
            seg = art.iloc[stop1_idx:stop2_idx+1]
            # if (seg.longitude.iloc[1:].to_numpy() - seg.longitude.iloc[:-1].to_numpy()).min() < 0:
            #     continue
            
            # segments.append((seg, good))
            art = art.iloc[stop2_idx:]
            if (stop1, stop2) not in segments_by_stop:
                segments_by_stop[(stop1, stop2)] = []
            segments_by_stop[(stop1, stop2)].append((seg, good))
    for idx2 in tqdm(range(1,len(stops)), desc="processing segments"):
        idx1 = idx2-1
        stop1 = stops[idx1]
        stop1_loc = stoplocations[stop1]
        stop1_name = stopnames[stop1]
        stop2 = stops[idx2]
        stop2_name = stopnames[stop2]
        stop2_loc = stoplocations[stop2]
        segments = segments_by_stop[(stop1, stop2)]

        # print(len(segments))
        trip_duration_by_stops = {'invalid': []}
        for s, valid in segments:
            stop_locs, seg_dur = full_stop_info(s, 5)
            if valid:
                nstops = len(stop_locs)
                if nstops not in trip_duration_by_stops:
                    trip_duration_by_stops[nstops] = []
                trip_duration_by_stops[nstops].append( {'triptime': seg_dur, 'stops': stop_locs})
            else:
                trip_duration_by_stops['invalid'].append({'triptime': seg_dur, 'stops': stop_locs})
        segment_info[(stop1, stop2)] = trip_duration_by_stops
    #     segment_stops = []
    #     for s in segments:
    #         nstops = count_stops(s.speed_mph, 5)
    #         if nstops != 3:
    #             continue
    #         segment_stops.append((s, where_stops(s,5)[1]))
        if len(trip_duration_by_stops) == 0:
            # breakpoint()
            continue
        ag = autocluster(np.array([ll2euclid(s['lat'], s['lng']) for stop_locs in trip_duration_by_stops.values() \
                        for stop_loc in stop_locs for s in stop_loc['stops']]))
    #     print(f'Traveling from {stop1_name} to {stop2_name}')
        j=0
        cluster_centers = [np.zeros(3) for _ in range(max(ag.labels_)+1)]

        for stop_locs in trip_duration_by_stops.values():
            for stop_loc in stop_locs:
                for s in stop_loc['stops']:
                    cluster_centers[ag.labels_[j]] += [s['lat'], s['lng'], 1]
                    j+=1 
        
        cluster_centers = [ (a/c, b/c) for a,b,c in cluster_centers]

        j=0
        for stop_locs in trip_duration_by_stops.values():
            for stop_loc in stop_locs:
                for s in stop_loc['stops']:
                    if geopy.distance.geodesic( (s['lat'], s['lng']), cluster_centers[ag.labels_[j]]).feet < 300:
                        s['label'] = ag.labels_[j]
                    else:
                        s['label'] = -1
                    j+=1
        for i in range(ag.n_clusters_):
            durations = []
            lat = 0
            lng = 0
            for j,tripdur_info in enumerate(trip_duration_by_stops.get(1,[])):
                if tripdur_info['stops'][0]['label'] == i:
                    lat += tripdur_info['stops'][0]['lat']
                    lng += tripdur_info['stops'][0]['lng']
                    durations.append( tripdur_info['triptime'])
            if len(durations) == 0:
                continue
            lat /= len(durations)
            lng /= len(durations)
            stop_count = (ag.labels_ == i).sum()
            trip_count = sum([len(t) for t in trip_duration_by_stops.values()])
            segment_stop_infos.append( {'lat': lat, 'lng': lng, 'durations': durations, \
                                        'nonstop_durations': [ t['triptime'] for t in trip_duration_by_stops.get(0,{})], \
                                        'stop1': stop1, 'stop2': stop2, \
                                        'count': stop_count, 'trip_count': trip_count})
        #     for n,info in sorted(trip_duration_by_stops.items(), key=lambda x: x[0]):
    return segment_stop_infos

def station_analysis(arts, stops):
    ## this time just extract the parts where it is near a station
    art_stops = []
    # speed_limit = 5
    
    stop_euclid_locs = [ ll2euclid(stoplocations[s]['lat'], stoplocations[s]['lng']) for s in stops]
    for idx, art in enumerate(tqdm(arts)):
    #     art = df[df.trip_id == artid]
        art = art.iloc[np.argsort(art.msg_time)]
        prev_speed = 100
        lat = 0
        lng = 0
        stop_start = -1
        duration = 0
        for _,row in art.iterrows():
            dists = np.linalg.norm(stop_euclid_locs - ll2euclid(row.latitude,row.longitude), axis=1)
            if  100 >= min(dists):
    #             if prev_speed < speed_limit:
                    duration_delta = str2time(row.msg_time) - stop_start
                    duration += duration_delta
                    lat += duration_delta * row.latitude
                    lng += duration_delta * row.longitude
                    station_idx = np.argmin(dists)
                    station_dist = min(dists)
    #                 stop_start = str2time(row.msg_time)
    #                 continue
    #             elif duration > 0:
    #                 stop_start = str2time(row.msg_time)
    #                 stop_location = (lat / duration, lng/duration)
                    
    # #                 prev_speed = 0
                    
    #                 duration = 0
    #                 lat = 0
    #                 lng = 0
            else:
                if duration > 0:
                    art_stops.append(((lat / duration, lng/duration), duration, idx, station_idx, station_dist))
                    lat = 0
                    lng = 0
                    duration = 0
                    
                    
            stop_start = str2time(row.msg_time)
            prev_speed = row.speed_mph
    art_stop_info = {}
    prev_arts = {}
    for a in tqdm(art_stops):
        lat = a[0][0]
        lng = a[0][1]
        duration = a[1]
        station = a[3]
        dist_to_station = a[4]
        if (a[2], station) not in art_stop_info or art_stop_info[(a[2], station)]['dist_to_station'] > a[4]:
            art_stop_info[(a[2], station)] = dict(lat=lat, lng=lng, duration=duration, dist_to_station=a[4], id=a[2], station=station)
    far_stops = list(art_stop_info.values())
    # np.random.shuffle(far_stops)
    # far_stops_euclid = [ ll2euclid(d['lat'], d['lng'])  for d in far_stops]
    # ag = autocluster(far_stops_euclid[:20000])
    ag = lambda: None
    ag.labels_ = np.array([a['station'] for a in far_stops])
    ag.n_clusters_ = len(stops)
    cluster_groups = cluster_grouper(ag, far_stops)
    max_radius = 300
    clusters = []
    for i,gp in enumerate(tqdm(cluster_groups)):
        if len(gp) == 0:
            continue
        lat = np.mean([g['lat'] for g in gp])
        lng = np.mean([g['lng'] for g in gp])
        rads = [ np.linalg.norm(ll2euclid(lat, lng) - ll2euclid(g['lat'], g['lng'])) for g in gp]
        lat = np.mean([g['lat'] for g,r in zip(gp, rads) if r < max_radius])
        lng = np.mean([g['lng'] for g,r in zip(gp, rads) if r < max_radius])
        duration = sum([g['duration'] for g,r in zip(gp, rads) if r < max_radius])
        durations = [g['duration'] for g,r in zip(gp, rads) if r < max_radius]
        station = set([g['station'] for g in gp])
        if len(station) != 1:
            breakpoint()
        station = list(station)[0]
        
    #     radius = max([r for g,r in zip(gp, rads) if r < max_radius])
        counts = len(set([g['id'] for g,r in zip(gp, rads) if r < max_radius]))
        clusters.append( dict(lat=lat, lng=lng, duration=duration, counts=counts, durations=durations, station=stops[station]))#, radius=radius))
    return clusters

def main(line_id):
    line_id = str(line_id)
    full_days = '10_11 10_10 10_8 10_9 9_28 9_23 9_21 9_22 9_12 10_7 9_19 9_16 9_17 9_18'.split()
    all_arts = []
    for day in tqdm(full_days, desc='Loading data'):
        with zipfile.ZipFile('../busdata/bus.zip') as file:
            with file.open(f'bus_data_2022_{day}.csv') as myfile:
                data = myfile.read()
        

        df = pd.read_csv(StringIO(data.decode()))
        arts = df[df.route_short_name==line_id]
        for id in set(arts.trip_id):
            all_arts.append(arts[arts.trip_id == id]) 
            all_arts[-1] = all_arts[-1].sort_values('msg_time', key=lambda x: [str2time(y) for y in x])


    # plt.imshow(dist_mat[order][:,order])
    # plt.show()    

    # filter by how close they are to real stops
    with open('stoplookup.json') as file:
        tripstops = json.load(file)
    # tripstops
    # filter by how close they are to real stops
    real_artstop_locations = [ stoplocations[s] for s,v in tqdm(tripstops.items()) if int(line_id) in v]
    real_artstop_locations

    allstops = set()
    for art in tqdm(all_arts):
        allstops = allstops.union(art.next_stop_id.tolist())
    allstops = list(allstops)
    allstop_locations = np.array([ll2euclid(stoplocations[i]['lat'], stoplocations[i]['lng']) for i in allstops if i in stoplocations]) # N x 2

    _,_,vt = np.linalg.svd(allstop_locations - np.mean(allstop_locations, axis=0), full_matrices=False)
    principal_axis = vt[0]

    if abs(principal_axis[0]) > abs(principal_axis[1]):
        print('North/South') # positive is northbound
        directions = 'north south'.split()
        principal_axis /= principal_axis[0]
    else:
        print('East/West') # positive is eastbound
        directions = 'east west'.split()
        principal_axis /= principal_axis[1]
    # ok, now we do a different analysis: counterfactuals
    all_arts_eb = [art.sort_values('msg_time', key=lambda x: [str2time(y) for y in x], kind='stable') for art in all_arts if (ll2euclid(art.iloc[-1].latitude, art.iloc[-1].longitude) - ll2euclid(art.iloc[0].latitude, art.iloc[0].longitude))@principal_axis > 0 ]
    all_arts_wb = [art.sort_values('msg_time', key=lambda x: [str2time(y) for y in x], kind='stable') for art in all_arts if (ll2euclid(art.iloc[-1].latitude, art.iloc[-1].longitude) - ll2euclid(art.iloc[0].latitude, art.iloc[0].longitude))@principal_axis <= 0 ]
    print(len(all_arts_eb), f' {directions[0]}bound')
    print(len(all_arts_wb), f' {directions[1]}bound')

    valid_arts_eb, eb_stops = filter_trips(all_arts_eb)
    valid_arts_wb, wb_stops = filter_trips(all_arts_wb)

    print(len(valid_arts_eb), f' {directions[0]}bound')
    print(len(valid_arts_wb), f' {directions[1]}bound')

    eb_station_info = station_analysis(valid_arts_eb, eb_stops)
    wb_station_info = station_analysis(valid_arts_wb, wb_stops)

    eb_stop_info = stop_analysis(valid_arts_eb, eb_stops)
    wb_stop_info = stop_analysis(valid_arts_wb, wb_stops)

    eb_segment_info = segment_analysis(valid_arts_eb, eb_stops)
    wb_segment_info = segment_analysis(valid_arts_wb, wb_stops)


    np.savez(ops.savefile, eb_stop_info=eb_stop_info, wb_stop_info=wb_stop_info, 
                            eb_station_info=eb_station_info, wb_station_info=wb_station_info, 
                            eb_segment_info=eb_segment_info, wb_segment_info=wb_segment_info,
                            eb_stops=eb_stops, wb_stops=wb_stops,
                            n_eb=len(valid_arts_eb), n_wb=len(valid_arts_wb))



if __name__ == '__main__':
    ops = options()
    main(ops.line_id)