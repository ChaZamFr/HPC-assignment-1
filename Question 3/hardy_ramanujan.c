#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef long long i64;

typedef struct {
    i64 sum;
    int a, b;
} SumEntry;

typedef struct {
    i64 sum;
    int a1,b1,a2,b2;
} HRNumber;

// ===== Function prototypes =====
double generate_seq(int n, int maxA, HRNumber* out_numbers);
double generate_par(int n, int maxA, int threads, int policy, HRNumber* out_numbers);
void save_hr_csv(const char* filename, HRNumber* numbers, int n);
void save_time_csv(const char* filename,int* threads,double* seq,double* stat,double* dyn,double* guid,double* task,double* speedup_stat,double* speedup_dyn,double* speedup_guid,double* speedup_task,int n);

// ===== Sequential HR generator =====
double generate_seq(int n, int maxA, HRNumber* out_numbers){
    double t0 = omp_get_wtime();
    int N = maxA;
    int total = (N*(N+1))/2;
    SumEntry* all = malloc(sizeof(SumEntry)*total);
    int pos = 0;
    for(int a=1;a<=N;a++){
        for(int b=a;b<=N;b++){
            all[pos++] = (SumEntry){.sum=(i64)a*a*a+(i64)b*b*b,.a=a,.b=b};
        }
    }
    int cmp_sum(const void* x,const void* y){
        i64 A = ((SumEntry*)x)->sum;
        i64 B = ((SumEntry*)y)->sum;
        return (A<B)?-1:(A>B)?1:0;
    }
    qsort(all,total,sizeof(SumEntry),cmp_sum);

    int res_count=0;
    for(int i=0;i<total && res_count<n;){
        int j=i+1;
        while(j<total && all[j].sum==all[i].sum) j++;
        if(j-i>=2){
            out_numbers[res_count].sum=all[i].sum;
            out_numbers[res_count].a1=all[i].a; out_numbers[res_count].b1=all[i].b;
            out_numbers[res_count].a2=all[i+1].a; out_numbers[res_count].b2=all[i+1].b;
            res_count++;
        }
        i=j;
    }
    free(all);
    return omp_get_wtime()-t0;
}

// ===== Parallel HR generator =====
double generate_par(int n, int maxA, int threads, int policy, HRNumber* out_numbers){
    double t0=omp_get_wtime();
    omp_set_num_threads(threads);

    int N = maxA;
    SumEntry** local_buffers = malloc(sizeof(SumEntry*)*threads);
    int* buf_sizes = malloc(sizeof(int)*threads);
    int* buf_caps = malloc(sizeof(int)*threads);

    for(int i=0;i<threads;i++){
        buf_caps[i]=N;
        buf_sizes[i]=0;
        local_buffers[i]=malloc(sizeof(SumEntry)*buf_caps[i]);
    }

    if(policy==4){ // Task-based
        #pragma omp parallel
        {
            #pragma omp single
            for(int a=1;a<=N;a++){
                #pragma omp task firstprivate(a)
                {
                    int tid = omp_get_thread_num();
                    for(int b=a;b<=N;b++){
                        int* size_ptr=&buf_sizes[tid];
                        if(*size_ptr>=buf_caps[tid]){
                            buf_caps[tid]*=2;
                            local_buffers[tid]=realloc(local_buffers[tid],sizeof(SumEntry)*buf_caps[tid]);
                        }
                        local_buffers[tid][(*size_ptr)++] = (SumEntry){.sum=(i64)a*a*a+(i64)b*b*b,.a=a,.b=b};
                    }
                }
            }
        }
    } else { // OpenMP for with static/dynamic/guided
        omp_sched_t sched_type;
        if(policy==1) sched_type=omp_sched_static;
        else if(policy==2) sched_type=omp_sched_dynamic;
        else sched_type=omp_sched_guided;
        omp_set_schedule(sched_type,16);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            SumEntry* buf = local_buffers[tid];
            int* size_ptr = &buf_sizes[tid];

            #pragma omp for schedule(runtime)
            for(int a=1;a<=N;a++){
                for(int b=a;b<=N;b++){
                    if(*size_ptr>=buf_caps[tid]){
                        buf_caps[tid]*=2;
                        buf=realloc(buf,sizeof(SumEntry)*buf_caps[tid]);
                        local_buffers[tid]=buf;
                    }
                    buf[(*size_ptr)++] = (SumEntry){.sum=(i64)a*a*a+(i64)b*b*b,.a=a,.b=b};
                }
            }
        }
    }

    // Merge buffers
    int total=0; for(int i=0;i<threads;i++) total+=buf_sizes[i];
    SumEntry* all = malloc(sizeof(SumEntry)*total);
    int pos=0;
    for(int i=0;i<threads;i++){
        for(int j=0;j<buf_sizes[i];j++) all[pos++]=local_buffers[i][j];
        free(local_buffers[i]);
    }
    free(local_buffers); free(buf_sizes); free(buf_caps);

    int cmp_sum(const void* x,const void* y){
        i64 A = ((SumEntry*)x)->sum;
        i64 B = ((SumEntry*)y)->sum;
        return (A<B)?-1:(A>B)?1:0;
    }
    qsort(all,total,sizeof(SumEntry),cmp_sum);

    int res_count=0;
    for(int i=0;i<total && res_count<n;){
        int j=i+1;
        while(j<total && all[j].sum==all[i].sum) j++;
        if(j-i>=2){
            out_numbers[res_count].sum=all[i].sum;
            out_numbers[res_count].a1=all[i].a; out_numbers[res_count].b1=all[i].b;
            out_numbers[res_count].a2=all[i+1].a; out_numbers[res_count].b2=all[i+1].b;
            res_count++;
        }
        i=j;
    }

    free(all);
    return omp_get_wtime()-t0;
}

// ===== Save HR numbers =====
void save_hr_csv(const char* filename, HRNumber* numbers,int n){
    FILE* fout=fopen(filename,"w");
    fprintf(fout,"Number,a^3,b^3,c^3,d^3\n");
    for(int i=0;i<n;i++)
        fprintf(fout,"%lld,%d,%d,%d,%d\n",numbers[i].sum,numbers[i].a1,numbers[i].b1,numbers[i].a2,numbers[i].b2);
    fclose(fout);
}

// ===== Save timing CSV with speedups =====
void save_time_csv(const char* filename,int* threads,double* seq,double* stat,double* dyn,double* guid,double* task,double* speedup_stat,double* speedup_dyn,double* speedup_guid,double* speedup_task,int n){
    FILE* fout=fopen(filename,"w");
    fprintf(fout,"Threads,Sequential,Static,Dynamic,Guided,Task,Speedup_Static,Speedup_Dynamic,Speedup_Guided,Speedup_Task\n");
    for(int i=0;i<n;i++){
        fprintf(fout,"%d,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf\n",
            threads[i],seq[i],stat[i],dyn[i],guid[i],task[i],speedup_stat[i],speedup_dyn[i],speedup_guid[i],speedup_task[i]);
    }
    fclose(fout);
}

// ===== Main =====
int main(int argc,char** argv){
    if(argc<4){printf("Usage: %s n maxA thread1 thread2 ...\n",argv[0]); return 1;}
    int n_hr = atoi(argv[1]);
    int maxA = atoi(argv[2]);
    int num_threads = argc-3;
    int* thread_list = malloc(sizeof(int)*num_threads);
    for(int i=0;i<num_threads;i++) thread_list[i] = atoi(argv[i+3]);

    HRNumber* hr_numbers = malloc(sizeof(HRNumber)*n_hr);

    double* seq_times = malloc(sizeof(double)*num_threads);
    double* stat_times = malloc(sizeof(double)*num_threads);
    double* dyn_times  = malloc(sizeof(double)*num_threads);
    double* guid_times = malloc(sizeof(double)*num_threads);
    double* task_times = malloc(sizeof(double)*num_threads);

    double* speedup_stat = malloc(sizeof(double)*num_threads);
    double* speedup_dyn  = malloc(sizeof(double)*num_threads);
    double* speedup_guid = malloc(sizeof(double)*num_threads);
    double* speedup_task = malloc(sizeof(double)*num_threads);

    for(int i=0;i<num_threads;i++){
        int t = thread_list[i];

        seq_times[i]  = generate_seq(n_hr,maxA,hr_numbers);
        stat_times[i] = generate_par(n_hr,maxA,t,1,hr_numbers);
        dyn_times[i]  = generate_par(n_hr,maxA,t,2,hr_numbers);
        guid_times[i] = generate_par(n_hr,maxA,t,3,hr_numbers);
        task_times[i] = generate_par(n_hr,maxA,t,4,hr_numbers);

        speedup_stat[i] = seq_times[i]/stat_times[i];
        speedup_dyn[i]  = seq_times[i]/dyn_times[i];
        speedup_guid[i] = seq_times[i]/guid_times[i];
        speedup_task[i] = seq_times[i]/task_times[i];

        printf("Threads: %d | Seq: %.6lf | Static: %.6lf | Dynamic: %.6lf | Guided: %.6lf | Task: %.6lf\n",
            t,seq_times[i],stat_times[i],dyn_times[i],guid_times[i],task_times[i]);
    }

    save_hr_csv("hr_numbers.csv",hr_numbers,n_hr);
    save_time_csv("hr_times.csv",thread_list,seq_times,stat_times,dyn_times,guid_times,task_times,speedup_stat,speedup_dyn,speedup_guid,speedup_task,num_threads);

    printf("HR numbers saved to hr_numbers.csv\nTiming saved to hr_times.csv\n");

    free(thread_list); free(hr_numbers);
    free(seq_times); free(stat_times); free(dyn_times); free(guid_times); free(task_times);
    free(speedup_stat); free(speedup_dyn); free(speedup_guid); free(speedup_task);

    return 0;
}
