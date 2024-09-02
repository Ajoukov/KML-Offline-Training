#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#include <cstdlib> // For strtof

using namespace std;
using namespace __gnu_pbds;

#define ll long long
#define ld long double
#define all(a) (a).begin(), (a).end()

const int MAX_N = 2e5 + 5;
const int MAX_Y = 1e9 + 5;

typedef tree<tuple<ld, ld, ld>, null_type, less<tuple<ld, ld, ld>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;

const int n = 34936;
const int sz = 16;
ld preds[n];
ld trues[n];

ld shape_preds[sz];
ld shape_perfs[sz];
ld shape_rands[sz];

double calculateAverageSlope(ld *y_values) {
    double totalSlope = 0.0;

    for (int i = 0; i < sz - 1; ++i) {
        double slope = (y_values[i + 1] - y_values[i]);
        totalSlope += slope;
    }

    double averageSlope = totalSlope / (sz - 1);
    return averageSlope;
}

int main(int argc, char* argv[]) {
    ios_base::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    
    FILE* file = freopen("predictions_simulate.csv", "r", stdin);
    for (int i = 0; i < n; i++) cin >> preds[i];
    for (int i = 0; i < n; i++) preds[i] = exp((ld)preds[i]);
    for (int i = 0; i < n; i++) if (preds[i] > 1e6) cout << "preds: " << i << "\n";
    fclose(file);

    file = freopen("latencies_simulate.csv", "r", stdin);
    for (int i = 0; i < n; i++) cin >> trues[i];
    for (int i = 0; i < n; i++) trues[i] = exp((ld)trues[i]);
    for (int i = 0; i < n; i++) if (trues[i] > 1e6) cout << "trues: " << i << "\n";
    fclose(file);

    ordered_set zipped_preds;
    ordered_set zipped_perfs;
    ordered_set zipped_rands;
    
    fill(shape_preds, shape_preds + sz, 0);
    fill(shape_perfs, shape_perfs + sz, 0);
    fill(shape_rands, shape_rands + sz, 0);

    char* end;
    ld c = 0.05;
    if (argc > 1) {
        c = strtof(argv[1], &end);
    }

    for (int i = 0; i < 15; i++) {
        zipped_preds.insert({preds[i] + i * c,  trues[i],   i});
        zipped_perfs.insert({trues[i] + i * c,  trues[i],   i});
        zipped_rands.insert({i,                 trues[i],   i});
    }

    for (int i = sz; i < n; i++) {
        zipped_preds.insert({preds[i] + i * c,  trues[i],   i});
        zipped_perfs.insert({trues[i] + i * c,  trues[i],   i});
        zipped_rands.insert({i,                 trues[i],   i});

        // check new shape

        for (int j = 0; j < sz; j++) {
            shape_preds[j] += get<1>(*zipped_preds.find_by_order(j));
            shape_perfs[j] += get<1>(*zipped_perfs.find_by_order(j));
            shape_rands[j] += get<1>(*zipped_rands.find_by_order(j));
            if (shape_preds[j] > n * 50000) cout << i << " " << get<1>(*zipped_preds.find_by_order(j)) << " " << shape_preds[j] << "\n", exit(0);
            if (shape_perfs[j] > n * 50000) cout << i << " " << get<1>(*zipped_perfs.find_by_order(j)) << " " << shape_perfs[j] << "\n", exit(0);
            if (shape_rands[j] > n * 50000) cout << i << " " << get<1>(*zipped_rands.find_by_order(j)) << " " << shape_rands[j] << "\n", exit(0);
        }
        
        zipped_preds.erase(zipped_preds.find_by_order(0));
        zipped_perfs.erase(zipped_perfs.find_by_order(0));
        zipped_rands.erase(zipped_rands.find_by_order(0));
    }
    
    printf("  rands     preds     perfs\n");
    char buffer[2048];
    int start = 0;
    for (int j = 0; j < sz; j++) {
        snprintf(&buffer[start], sizeof(buffer), "%6.4Lf  %6.4Lf  %6.4Lf\n", shape_rands[j] / (n-sz), shape_preds[j] / (n-sz), shape_perfs[j] / (n-sz));
        start += strlen(&buffer[start]);
    }
    snprintf(&buffer[start], sizeof(buffer), "\n\n%6.4f  %6.4f  %6.4f\n", calculateAverageSlope(shape_rands), calculateAverageSlope(shape_preds), calculateAverageSlope(shape_perfs));
    printf("%s\n", buffer);
}