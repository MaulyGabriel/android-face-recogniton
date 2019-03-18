package org.bytedeco.javacv.android.recognize.example;

public class Operator {

    int code;
    String nome;

    public Operator(int code, String nome) {
        this.code = code;
        this.nome = nome;
    }

    public int getCode() {
        return code;
    }

    public void setCode(int code) {
        this.code = code;
    }

    public String getNome() {
        return nome;
    }

    public void setNome(String nome) {
        this.nome = nome;
    }

}
