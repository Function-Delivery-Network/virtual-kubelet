FROM --platform=${TARGETPLATFORM:-linux/amd64} ghcr.io/openfaas/of-watchdog:0.9.10  as watchdog
FROM --platform=${TARGETPLATFORM:-linux/amd64} isaacnez/base-handler:latest

ARG TARGETPLATFORM
ARG BUILDPLATFORM

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

ARG ADDITIONAL_PACKAGE
RUN apt-get install -y --no-install-recommends musl-dev gcc make ${ADDITIONAL_PACKAGE}

# Add non root user
RUN groupadd -r app && useradd -r -g app app
RUN mkdir -p /home/app
RUN chown app /home/app

USER app

ENV PATH=$PATH:/home/app/.local/bin

WORKDIR /home/app/

# COPY requirements.txt   .
# USER root
# RUN pip install -r requirements.txt
USER app
COPY index.py           .

RUN mkdir -p function
RUN touch ./function/__init__.py
WORKDIR /home/app/function/
COPY function/requirements.txt	.
RUN pip install --user -r requirements.txt

WORKDIR /home/app/

USER root
COPY function function
RUN chown -R app:app ./
USER app

ENV fprocess="uvicorn index:app --host 0.0.0.0 --port 9000"

ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:9000"
ENV exec_timeout=600s
ENV write_timeout=300s
ENV read_timeout=300s

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]