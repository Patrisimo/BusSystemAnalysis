import numpy as np
import urllib
import json
from argparse import ArgumentParser
import geopy.distance

with open('stoplocations.json') as file:
    stoplocations = json.load(file)
    
with open('stopnames.json') as file:
    stopnames = json.load(file)    

def options():
    parser = ArgumentParser()
    parser.add_argument('datafile')

    return parser.parse_args()


def elbow(data):
    elbowdata = np.array(data)
    if elbowdata.shape[1] != 2:
        elbowdata = elbowdata.T
    elbowdata -= elbowdata[0]
    rotation = np.linalg.inv(np.vstack([elbowdata[-1], [-elbowdata[-1,1], elbowdata[-1,0]]]).T)
    idx = np.argmin((elbowdata @rotation.T)[:,1])
    return idx  

recent_searches = {}
interstring = "http://api.geonames.org/findNearestIntersectionOSMJSON?lat={}&lng={}&username=patrisimo"
def intersection_label(lat, lng):
    stop_distances = [ (id, geopy.distance.geodesic((lat,lng),(x['lat'], x['lng'])).feet) for id,x in stoplocations.items()]
    closest_stop, d = min(stop_distances, key=lambda x: x[1])
    if d > 20:
        stop_distances = [ (name, geopy.distance.geodesic((lat,lng),x).feet) for x,name in recent_searches.items()]
        closest_stop, d = min(stop_distances, key=lambda x: x[1], default=(None, 1000))
        if d < 20:
            return closest_stop, None
        else:
            lookup = json.loads( urllib.request.urlopen(interstring.format(lat, lng)).read().decode())
            name = f"{lookup['intersection']['street1']} & {lookup['intersection']['street2']}"
            d = geopy.distance.geodesic((lat,lng),(lookup['intersection']['lat'], lookup['intersection']['lng'])).feet
            if d < 200:
                recent_searches[(lat,lng)] = name
                return name, None
            else:
                return f'{lat}, {lng}', None

    else:
        return stopnames[closest_stop], closest_stop

def print_segment_info(segment_stop_infos, eb_stops):
    last_pair = None
    sorted_clusters = sorted(segment_stop_infos, key=lambda x: (-np.mean(x["durations"]) + np.mean(x["nonstop_durations"]))*x["count"]/x["trip_count"] )
    excess_wait = [(np.mean(x["durations"]) - np.mean(x["nonstop_durations"]))*x["count"]/x["trip_count"]  for x in sorted_clusters]
    excess_wait_pct = np.cumsum(excess_wait) / sum(excess_wait)
    print("Total expected wait: ", round(sum(excess_wait),3), 's')
    ct = min(len(excess_wait_pct)-1, min(10, np.argmax(excess_wait_pct > 0.95)+1))
    print(f"Worst {ct} stations: {round(excess_wait_pct[ct]*100,2)}% of wait")        
    for ssi in sorted_clusters[:ct]:
        stop1_name = stopnames[ssi['stop1']]
        stop2_name = stopnames[ssi['stop2']]
        if stop1_name != last_pair:
            print(f'{stop1_name} to {stop2_name}')
        last_pair = stop1_name
        print(f'\tStop only at {intersection_label(ssi["lat"],ssi["lng"])[0]}: ({len(ssi["durations"])} many)')
        print(f'\t{ssi["lat"]}, {ssi["lng"]}')
        # print(f'\t\tAverage delay: {np.mean(ssi["durations"]) - np.mean(ssi["nonstop_durations"])}s')
        # print(f'\t\tMedian delay: {np.median(ssi["durations"]) - np.median(ssi["nonstop_durations"])}s')
        print(f'\t\tProbability: {round(ssi["count"]*100/ssi["trip_count"], 2)}%')
        print(f'\t\tExpected delay: {round((np.mean(ssi["durations"]) - np.mean(ssi["nonstop_durations"]))*ssi["count"]/ssi["trip_count"], 2)}s')

def print_segment_analysis(segment_stop_infos, eb_stops):
    
    for i,stop2 in enumerate(eb_stops[1:]):
        stop1 = eb_stops[i]
        stop1_name = stopnames[stop1]
        stop2_name = stopnames[stop2]    
        segs = [ ssi for ssi in segment_stop_infos if ssi['stop1'] == stop1 and ssi['stop2'] == stop2]
        print(f'{stop1_name} - {stop2_name}')
        average_travel_time = np.mean([d for x in segs for d in x['durations']])
        average_nonstop_travel_time = np.mean([d for x in segs for d in x['nonstop_durations']])
        print(f'\tAverage duration: {average_travel_time}')
        print(f'\tNonstop duration: {average_nonstop_travel_time}')
        print(f'\tTime lost:        {average_travel_time - average_nonstop_travel_time}')


def print_station_analysis(clusters, eb_stops, n_buses):
    durs = sorted([c for cl in clusters for c in cl['durations']])
    idx = elbow(list(enumerate(durs)))
    idx2 = elbow(list(enumerate(durs[:idx])))
    wait_limit = durs[idx]
    # wait_allowance = durs[idx2] 
    wait_allowance = 72
    for idx,stop in enumerate(eb_stops):
        stop_name = stopnames[stop]
        cluster = [c for c in clusters if c['station'] == stop]
        if len(cluster) == 0:
            print(f'No info for {stop_name}')
            continue
        if len(cluster) != 1:
            print('oops')
            continue
        cluster = cluster[0]
        print(stop_name)
        durs = cluster['durations']
        # print('\tAverage wait: ', np.mean(durs))
        # print(f'\tAverage wait (<{round(wait_limit, 3)}): ', np.mean([d for d in durs if d < wait_limit]))
        # print(f'\tAverage wait ({round(wait_allowance, 3)} cap): ', np.mean([min(wait_allowance, d) for d in durs if d < wait_limit]))   
        # print(f'\tTime lost: ', np.mean([d - min(wait_allowance, d) for d in durs if d < wait_limit]))  
        print(f'\tProbability of stop: {round(len(durs)*100/n_buses,3)}% ({len(durs)}/{n_buses})')
        print(f'\tExpected time loss:', np.mean([d - min(wait_allowance, d) for d in durs if d < wait_limit])*len(durs)/n_buses)   

def print_station_info(clusters, n_buses):
    durs = sorted([c for cl in clusters for c in cl['durations']])
    # breakpoint()
    idx = elbow(list(enumerate(durs)))
    idx2 = elbow(list(enumerate(durs[:idx])))
    wait_limit = durs[idx]
    wait_allowance = durs[idx2]
    print('Wait limit: ', wait_limit)
    print('Wait allowance: ', wait_allowance)
    print(f'Average wait: ', np.mean([d for d in durs]))
    print(f'Average wait (<{round(wait_limit, 3)}): ', np.mean([d for d in durs if d < wait_limit]))
    print(f'Average wait ({round(wait_allowance, 3)} cap): ', np.mean([min(wait_allowance, d) for d in durs if d < wait_limit]))
    sorted_clusters = sorted(clusters, key=lambda x: -sum([ max(0,c-wait_allowance) for c in x['durations'] if c < wait_limit]))
    excess_wait = [sum([ max(0,c-wait_allowance) for c in x['durations'] if c < wait_limit]) for x in sorted_clusters]
    excess_wait_pct = np.cumsum(excess_wait) / sum(excess_wait)
    print("Total expected excess wait: ", round(sum(excess_wait)/n_buses,3), 's')
    ct = min(len(excess_wait_pct)-1, min(10, np.argmax(excess_wait_pct > 0.95)+1))
    print(f"Worst {ct} stations: {round(excess_wait_pct[ct]*100,2)}% of wait")    

    for i,cluster in enumerate(sorted_clusters[:ct]):
        if cluster['counts'] == 0:
            continue
        print('Cluster ', i+1)
        
        print('\t',intersection_label(cluster['lat'], cluster['lng'])[0])
        print('\tLat/lon: ', round(cluster['lat'],8), ', ', round(cluster['lng'],8))
        # print('\tTotal delay: ', cluster['duration']/60, 'min')
        # print(f'\tTotal delay (<{round(wait_limit/60,3)} minutes): ', round(sum([c for c in cluster['durations'] if c < wait_limit])/60,3))
        # print('\tNumber of stops: ', cluster['counts'], f'({round(cluster["counts"]*100/n_buses, 2)}%)')
        # print('\tAverage delay: ', sum([c for c in cluster['durations'] if c < wait_limit]) / (60 * cluster['counts']), 'min')
        print('\tExpected excess delay: ', round(sum([ max(0,c-wait_allowance) for c in cluster['durations'] if c < wait_limit])/n_buses,3), 's')
        print('\tProb of excess delay: ', round(100*sum([ 1 if c > wait_allowance else 0 for c in cluster['durations'] if c < wait_limit])/n_buses,3), '%')
        # print(f'\t\t25%ile: {round(np.quantile([c for c in cluster["durations"] if c < wait_limit], 0.25)/60, 3)}')
        # print(f'\t\t50%ile: {round(np.quantile([c for c in cluster["durations"] if c < wait_limit], 0.5)/60, 3)}')
        # print(f'\t\t75%ile: {round(np.quantile([c for c in cluster["durations"] if c < wait_limit], 0.75)/60, 3)}')
        print()

def print_stop_info(clusters, n_buses):
    ds = sorted([ d for c in clusters for d in c['durations']])
    idx = elbow(list(enumerate(ds)))
    max_duration = ds[idx]
    sorted_clusters = sorted(clusters, key=lambda x: -sum([ c for c in x['durations'] if c < max_duration]))
    total_delay = [sum([ c for c in x['durations'] if c < max_duration]) for x in sorted_clusters]
    total_delay_pct = np.cumsum(total_delay) / sum(total_delay)
    print("Total expected delay: ", round(sum(total_delay)/n_buses,3), 's')
    ct = min(len(total_delay_pct)-1, min(10, np.argmax(total_delay_pct > 0.95)+1))
    print(f"Worst {ct} stops: {round(total_delay_pct[ct]*100,2)}% of delay")      
    for i,cluster in enumerate(sorted_clusters[:ct]):
        delay = sum([ c for c in cluster['durations'] if c < max_duration])
        print('Cluster ', i+1, f' - {intersection_label(cluster["lat"], cluster["lng"])[0]}')
        print('\tLat/lon: ', round(cluster['lat'],8), ', ', round(cluster['lng'],8))
        # print('\tTotal delay: ', delay/60, 'min')
        print('\tNumber of stops: ', cluster['counts'], f'({round(cluster["counts"]*100/n_buses, 2)}%)')
        # print('\tAverage delay: ', delay / (60 * cluster['counts']), 'min')
        print('\tExpected delay: ', delay / ( n_buses), 's')
        # print(f'\t\t25%ile: {round(np.quantile(cluster["durations"], 0.25)/60, 3)}')
        # print(f'\t\t50%ile: {round(np.quantile(cluster["durations"], 0.5)/60, 3)}')
        # print(f'\t\t75%ile: {round(np.quantile(cluster["durations"], 0.75)/60, 3)}')
        print()


def main(filename):
    data = np.load(filename, allow_pickle=True)
    print("East/North bound")
    print("Stops")
    print_stop_info(data['eb_stop_info'], data['n_eb'])
    print("Stations")
    print_station_info(data['eb_station_info'], data['n_eb'])
    print("Segments")
    print_segment_info(data['eb_segment_info'], data['eb_stops'])
    print("Segment analysis")
    print_segment_analysis(data['eb_segment_info'], data['eb_stops'])
    print("Station analysis")
    print_station_analysis(data['eb_station_info'], data['eb_stops'], data['n_eb'])
    print('--')
    print("West/South bound")
    print("Stops")
    print_stop_info(data['wb_stop_info'], data['n_wb'])
    print("Stations")
    print_station_info(data['wb_station_info'], data['n_wb'])
    print("Segments")
    print_segment_info(data['wb_segment_info'], data['wb_stops'])
    print("Segment analysis")
    print_segment_analysis(data['wb_segment_info'], data['wb_stops'])
    print("Station analysis")    
    print_station_analysis(data['wb_station_info'], data['wb_stops'], data['n_wb'])


if __name__ == '__main__':
    ops = options()
    try:
        with open('recent_lookups.json') as file:
            recent_lookups = json.load(file)
        for lat, lookup in recent_lookups.items():
            for lng, name in lookup.items():
                recent_searches[(lat,lng)] = name
    except:
        pass
    main(ops.datafile)
    with open('recent_lookups.json', 'w') as file:
        recent_lookups = {}
        for (lat,lng),name in recent_searches.items():
            if lat not in recent_lookups:
                recent_lookups[lat] = {}
            recent_lookups[lat][lng] = name
        json.dump(recent_lookups, file)