/* Sample from Devign Dataset - QEMU Xen 9pfs driver */
/* Label: VULNERABLE (target=1) - Memory leak, missing free on error path */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>

#define MAX_RINGS 8
#define MAX_RING_ORDER 9
#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define FD_CLOEXEC 1
#define F_SETFD 2

typedef struct Xen9pfsRing {
    void *priv;
    int evtchn;
    int local_port;
    int ref;
    void *intf;
    void *data;
    int ring_order;
    void *bh;
    int out_cons;
    int out_size;
    int inprogress;
    void *evtchndev;
    struct {
        void *in;
        void *out;
    } ring;
} Xen9pfsRing;

typedef struct Xen9pfsDev {
    int num_rings;
    Xen9pfsRing *rings;
    char *security_model;
    char *path;
    char *id;
    char *tag;
} Xen9pfsDev;

static int xen_9pfs_connect(Xen9pfsDev *xen_9pdev)
{
    int i;
    
    if (xen_9pdev->num_rings > MAX_RINGS || xen_9pdev->num_rings < 1) {
        return -1;
    }

    /* VULNERABILITY: Memory allocated but not freed on all error paths */
    xen_9pdev->rings = malloc(xen_9pdev->num_rings * sizeof(Xen9pfsRing));
    
    for (i = 0; i < xen_9pdev->num_rings; i++) {
        char *str;
        int ring_order;

        xen_9pdev->rings[i].priv = xen_9pdev;
        xen_9pdev->rings[i].evtchn = -1;
        xen_9pdev->rings[i].local_port = -1;

        /* VULNERABILITY: str is allocated but never freed - memory leak */
        str = malloc(32);
        sprintf(str, "ring-ref%u", i);
        
        /* Simulated read failure */
        if (str == NULL) {
            goto out;  /* Memory leak: str not freed */
        }

        /* VULNERABILITY: Another allocation without free */
        str = malloc(32);
        sprintf(str, "event-channel-%u", i);

        xen_9pdev->rings[i].intf = malloc(64);
        if (!xen_9pdev->rings[i].intf) {
            goto out;  /* Memory leak: previous allocations not freed */
        }

        ring_order = 4;
        if (ring_order > MAX_RING_ORDER) {
            goto out;
        }
        xen_9pdev->rings[i].ring_order = ring_order;

        xen_9pdev->rings[i].data = malloc(1 << ring_order);
        if (!xen_9pdev->rings[i].data) {
            goto out;  /* Memory leak */
        }

        xen_9pdev->rings[i].evtchndev = malloc(32);
        if (xen_9pdev->rings[i].evtchndev == NULL) {
            goto out;
        }

        /* VULNERABILITY: Using unvalidated file descriptor */
        fcntl(*(int*)xen_9pdev->rings[i].evtchndev, F_SETFD, FD_CLOEXEC);

        xen_9pdev->rings[i].local_port = -1;
        if (xen_9pdev->rings[i].local_port == -1) {
            printf("bind failed port=%d\n", xen_9pdev->rings[i].evtchn);
            goto out;
        }
    }

    return 0;

out:
    /* VULNERABILITY: Incomplete cleanup - not all allocated memory is freed */
    /* Missing: free of str, rings[i].intf, rings[i].data, rings[i].evtchndev */
    free(xen_9pdev->rings);
    return -1;
}

int main(void) {
    Xen9pfsDev dev;
    dev.num_rings = 2;
    
    int result = xen_9pfs_connect(&dev);
    printf("Connect result: %d\n", result);
    
    return 0;
}
